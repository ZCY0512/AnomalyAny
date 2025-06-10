import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
from run import get_indices_to_alter_new, run_on_prompt_and_masked_image
from utils.ptp_utils import AttentionStore
from config import RunConfig

NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5


def run_generation(stable, prompt, normal_prompt, detailed_prompt, tokens, image, mask_path, seed, scale_factor, guidance_scale):
    token_indices = get_indices_to_alter_new(stable, prompt, tokens)
    controller = AttentionStore()
    g = torch.Generator(device=stable.device).manual_seed(seed)
    config = RunConfig(
        prompt=prompt,
        run_standard_sd=False,
        scale_factor=scale_factor,
        max_iter_to_alter=25,
        n_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    result, _ = run_on_prompt_and_masked_image(
        prompt=[prompt],
        model=stable,
        controller=controller,
        token_indices=token_indices,
        init_image=image,
        init_image_guidance_scale=guidance_scale,
        mask_image=mask_path,
        seed=g,
        config=config,
        normal_prompt=normal_prompt,
        detailed_prompt=detailed_prompt,
    )
    return result


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stable = RelationalAttendAndExcitePipeline.from_pretrained(
        args.model_path, safety_checker=None).to(device)

    with open(args.prompts_file, 'r') as f:
        prompt_cfg = json.load(f)

    ok_images = sorted(Path(args.ok_dir).glob('*.*'))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in ok_images:
        img = Image.open(img_path).convert('RGB')
        base_name = img_path.stem
        for defect, cfg in prompt_cfg.items():
            prompt = cfg['prompt'].format(object=args.object_name)
            normal_prompt = cfg.get('normal_prompt', 'a photo of a {object}').format(object=args.object_name)
            detailed_prompt = cfg.get('detailed_prompt', prompt).format(object=args.object_name)
            tokens = cfg['tokens']
            mask = cfg.get('mask')
            if mask is not None:
                mask = str(Path(mask))
            result = run_generation(
                stable=stable,
                prompt=prompt,
                normal_prompt=normal_prompt,
                detailed_prompt=detailed_prompt,
                tokens=tokens,
                image=img,
                mask_path=mask,
                seed=args.seed,
                scale_factor=args.scale_factor,
                guidance_scale=args.guidance_scale,
            )
            out_dir = output_dir / defect
            out_dir.mkdir(parents=True, exist_ok=True)
            result.save(out_dir / f"{base_name}_{defect}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic defect images using AnomalyAny.')
    parser.add_argument('--ok-dir', required=True, help='Directory containing normal images.')
    parser.add_argument('--prompts-file', required=True, help='JSON file mapping defect names to prompts.')
    parser.add_argument('--model-path', default='runwayml/stable-diffusion-v1-5', help='Path or repo id of Stable Diffusion weights.')
    parser.add_argument('--output-dir', default='generated_defects', help='Directory to store generated images.')
    parser.add_argument('--object-name', default='object', help='Object name placeholder used in prompts.')
    parser.add_argument('--guidance-scale', type=float, default=0.3, help='Init image guidance scale.')
    parser.add_argument('--scale-factor', type=int, default=50, help='Latent update rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    main(args)
