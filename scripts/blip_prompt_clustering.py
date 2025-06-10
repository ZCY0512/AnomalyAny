import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import clip
from lavis.models import load_model_and_preprocess


def generate_captions(image_paths, device):
    """Generate captions for images using BLIP."""
    blip_model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    captions = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inp = vis_processors["eval"](image).unsqueeze(0).to(device)
        with torch.no_grad():
            caption = blip_model.generate({"image": inp})[0]
        captions.append(caption)
    return captions


def cluster_prompts(prompts, num_clusters, device):
    """Cluster prompts using CLIP text embeddings."""
    clip_model, _ = clip.load("ViT-B/16", device)
    clip_model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        embeddings = clip_model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    representatives = []
    for i in range(num_clusters):
        idxs = np.where(labels == i)[0]
        cluster_embs = embeddings[idxs]
        centroid = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        chosen_idx = idxs[np.argmin(dists)]
        representatives.append(prompts[chosen_idx])
    return representatives, labels.tolist()


def main(args):
    image_dir = Path(args.image_dir)
    image_paths = [p for p in image_dir.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Generating captions for {len(image_paths)} images...")
    captions = generate_captions(image_paths, device)

    print("Clustering prompts...")
    reps, labels = cluster_prompts(captions, args.num_clusters, device)

    result = {
        "captions": dict(zip([p.name for p in image_paths], captions)),
        "representative_prompts": reps,
        "labels": dict(zip([p.name for p in image_paths], labels)),
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Representative prompts:")
    for p in reps:
        print(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts using BLIP and cluster them.")
    parser.add_argument("--image-dir", required=True, help="Directory with defect images")
    parser.add_argument("--num-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--output-json", default="clustered_prompts.json", help="Where to save results")
    args = parser.parse_args()
    main(args)
