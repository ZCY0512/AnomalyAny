import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
# from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention
from clip_loss import CLIPLoss
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# TODO
from PIL import Image
from torch import autocast
import gc

logger = logging.get_logger(__name__)


def image2latent(vae, image, width, height, device, generator):
    init_image = image
    # Resize and transpose for numpy b h w c -> torch b c h w
    init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
    init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
    init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

    # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if init_image.shape[1] > 3:
        init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

    # Move image to GPU
    init_image = init_image.to(device)

    # Encode image
    with autocast(device):
        init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

    return init_latent


def read_mask(mask_path: str, dest_size=(64, 64)):
    if isinstance(mask_path, str):
        org_mask = Image.open(mask_path).convert("L")
    else: 
        org_mask = mask_path.convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask)
    # print(mask)
    mask[mask != 0] = 255
    mask = np.array(mask) / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask[np.newaxis, np.newaxis, ...]
    mask = torch.from_numpy(mask).half().to('cuda')

    return mask, org_mask


class RelationalAttendAndExcitePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def decode_latents_new(self, latents):
        # deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        self.vae.zero_grad()
        image = torch.clamp(image / 2 + 0.5, min=0, max=1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = image.permute(0, 2, 3, 1)
        return image

    def latent_process(self, img_latent):
        img_latent = img_latent.permute(0, 3, 1, 2)
        img_latent_resized = torch.nn.functional.interpolate(img_latent, size=(224, 224), mode='bicubic',
                                                             align_corners=False)
        transform = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_latent_resized = transform(img_latent_resized)
        return img_latent_resized

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            img_prompt=None,
            indice=None,
            normal_prompt=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            # print(prompt_embeds.shape) # torch.Size([1, 77, 768])
            # TODO switch prompt with image embedding
            # print(prompt[:, indice, :].shape)
            # if img_prompt is not None:
            #    prompt_embeds[:, indice, :] = prompt_embeds_normal[:, indice, :] + (prompt_embeds[:, indice, :] - prompt_embeds_normal[:, indice, :]).norm(dim=-1, keepdim=True)*img_prompt/img_prompt.norm(dim=-1, keepdim=True)
            # prompt_embeds[:, indice, :] = prompt_embeds[:, indice, :] - prompt_embeds_normal[:, indice, :] + img_prompt
            # prompt_embeds[:, indice, :] = prompt_embeds_normal[:, indice, :] + img_prompt
            # prompt_embeds[:, indice, :] = img_prompt

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        # print("before", prompt_embeds.size()) # [1, 77, 768]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # print("after", prompt_embeds.size())  # [1, 77, 768]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # print("final", prompt_embeds.size())  # [1, 77, 768]

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         return_attention: bool = False) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        if return_attention:
            return max_indices_list, image
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   return_maps: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index, attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            return_attention=True)
        if return_maps:
            return max_attention_per_index, attention_per_index
        return max_attention_per_index

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False,
                      compute_clip=True) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if return_losses:
            return loss, losses
        else:
            del losses
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, return_grad=False) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # print()
        latents = latents - step_size * grad_cond
        if return_grad:
            return latents, grad_cond
        del grad_cond
        del loss
        # gc.collect()
        # torch.cuda.empty_cache()
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        # while loss > target_loss:
        while iteration < max_refinement_steps:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
            )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                # print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                #   f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        # print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index
    
    def _perform_att(self,
                    latents: torch.Tensor,
                    indices_to_alter: List[int],
                    text_embeddings: torch.Tensor,
                    text_input,
                    attention_store: AttentionStore,
                    step_size: float,
                    t: int,
                    attention_res: int = 16,
                    smooth_attentions: bool = True,
                    sigma: float = 0.5,
                    kernel_size: int = 3,
                    max_refinement_steps: int = 20,
                    normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        # latents = latents.clone().detach().requires_grad_(True)
        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        # latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        # print(f"\t Finished with loss of: {loss}")
        return loss, max_attention_per_index
    
    def _prompt_update(self, prompt_embeds, prompt_embeds_original, normal_embeds, indices_to_alter, curr_step_size=0.1):
        criterion_cosine = torch.nn.CosineSimilarity()
        prompt_anomaly= prompt_embeds[:, indices_to_alter, :]
        for k in range(20):
            with torch.enable_grad():
                prompt_anomaly = prompt_anomaly.detach().requires_grad_(True)
                delta_1 = self._compute_dist(normal_embeds[:, indices_to_alter, :], prompt_anomaly)
                delta_2 = self._compute_dist(normal_embeds, prompt_embeds_original)
                loss_clip = criterion_cosine(delta_1, delta_2).mean()
                loss_prompt =  1.0 * (1.0 - loss_clip)
                # print(criterion_cosine(delta_1, delta_2))
                # print(loss_prompt)
                prompt_anomaly = self._update_latent(latents=prompt_anomaly, loss=loss_prompt, step_size=curr_step_size)
        prompt_embeds[:, indices_to_alter, :] = prompt_anomaly
        # print(loss_prompt)
        del loss_prompt
        return prompt_embeds
    
    def _compute_dist(self, t1, t2):
        delta = t2.mean(dim=0, keepdim=True) - t1.mean(dim=0, keepdim=True)
        # delta = delta/delta.norm(dim=-1, keepdim=True)
        delta_new = torch.nan_to_num(delta)

        return delta_new


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            init_image=None,
            init_image_guidance_scale: float = 0.5,
            mask_image=None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            img_prompt: torch.Tensor = None,
            normal_prompt=None,
            abnormal_img=None,
            original_prompt=None,
            detailed_prompt=None,
            clip_loss=None
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        criterion_mse = torch.nn.MSELoss()
        criterion_cosine = torch.nn.CosineSimilarity()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)
        model.train()
        clip_loss = CLIPLoss(device,
                             lambda_direction=1.0,
                             lambda_patch=0.0,
                             lambda_global=0.0,
                             lambda_manifold=0.0,
                             lambda_texture=0.0,
                             clip_model=model, clip_processor=preprocess)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # print("===========", indices_to_alter[0]) [8]

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=normal_prompt

        )
        prompt_original = detailed_prompt
        original_inputs, prompt_embeds_original = self._encode_prompt(
            [prompt_original],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=None
        )

        normal_inputs, normal_embeds = self._encode_prompt(
            [normal_prompt],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=None
        )
        #
        # print(normal_prompt, prompt, prompt_original)
        model.zero_grad()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        if init_image is not None:
            latents = image2latent(self.vae, init_image, width, height, "cuda", generator)

        num_channels_latents = self.unet.in_channels
        latents_source = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if mask_image is not None:
            latent_mask, org_mask = read_mask(mask_image)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Test-time Normal Sample Conditioning: init_image_guidance_scale
        if init_image is None:
            t_start = 0
        else:
            t_start = num_inference_steps - int(num_inference_steps * init_image_guidance_scale)
        timesteps_initial = timesteps
        timesteps = timesteps[t_start:]
        # print(timesteps)

        loss_img = 1000000
        k_range = 1
        clip_step_size = 0.02

        prompt_embeds = prompt_embeds.detach().requires_grad_(True)
        prompt_embeds = self._prompt_update(prompt_embeds=prompt_embeds,
                                            prompt_embeds_original=prompt_embeds_original,
                                            indices_to_alter=indices_to_alter,
                                            normal_embeds=normal_embeds)


        for k in range(k_range):
            #Generate random normal noise
            noise = torch.randn(latents.shape, generator=generator, device='cuda')
            # latent = noise * scheduler.init_noise_sigma
            latents = self.scheduler.add_noise(latents, noise,
                                               torch.tensor([self.scheduler.timesteps[t_start]], device='cuda')).to(
                'cuda')

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

            if max_iter_to_alter is None:
                max_iter_to_alter = len(self.scheduler.timesteps) + 1

            # 7. Denoising loop
            localization_update = True
            localization_count = 0
            n_start = 100
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # update with L_img, L_att and L_prompt
                    with torch.enable_grad():
                        
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents = latents.clone().detach().requires_grad_(True)
                        prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)
                        curr_step_size = max(0.0001, clip_step_size - 0.0001 * i)

                        # L = L_img + alpha*L_att
                        for q in range(10):

                            # L_img
                            if i > 0.7*(num_inference_steps-t_start):
                                image_curr = self.decode_latents_new(latents)
                                img_tensor = self.latent_process(image_curr)
                                loss_img = clip_loss.global_clip_loss(img_tensor, prompt_original)
                                loss_img = 1.0 * loss_img
                                del image_curr, img_tensor
                            else:
                                loss_img = 0

                            # L_att
                            # Get max activation value for each subject token
                            max_attention_per_index, maps_curr = self._aggregate_and_get_max_attention_per_token(
                                attention_store=attention_store,
                                indices_to_alter=indices_to_alter,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1,
                                return_maps=True)
                            maps_curr = maps_curr / torch.max(maps_curr)
                            maps_ = maps_curr.clone()
                            map_m = torch.mean(maps_curr)
                            maps_[maps_curr < map_m] = 0
                            maps_[maps_curr >= map_m] = 1
                            maps_curr = maps_
                            n_curr = torch.sum(maps_curr)
                            if i == 0:
                                n_start = n_curr
                            if 10 < n_curr < 50:
                                localization_update = False
                            loss_att, max_attention_per_index = self._perform_att(
                                        # TODO
                                        latents=latents,
                                        indices_to_alter=indices_to_alter,
                                        text_embeddings=prompt_embeds,
                                        text_input=text_inputs,
                                        attention_store=attention_store,
                                        step_size=n_curr/n_start*scale_factor * np.sqrt(scale_range[i]),
                                        t=t,
                                        attention_res=attention_res,
                                        smooth_attentions=smooth_attentions,
                                        sigma=sigma,
                                        kernel_size=kernel_size,
                                        normalize_eot=sd_2_1)
                            if i >= 0:
                                loss_img = 0.1*loss_img + 0.5*(i/(num_inference_steps-t_start))*loss_att
                                # update latent
                                if loss_img != 0 or loss_att != 0:
                                    latents = self._update_latent(latents=latents, loss=loss_img, step_size=curr_step_size*2)
                                # L_prompt
                                loss_prompt = loss_img + (1.0-criterion_cosine(prompt_embeds, prompt_embeds_original).mean())
                                # update embedding
                                prompt_embeds = self._update_latent(latents=prompt_embeds, loss=loss_prompt, step_size=curr_step_size)

                                del loss_img, loss_prompt

                            del max_attention_per_index
                            del maps_curr, maps_

                    del noise_pred_uncond, noise_pred_text, noise_pred, latent_model_input

                    
                    
                    
                    # additional attention optimization with L_att at early stage
                    with torch.enable_grad():

                        if i < 10 and localization_update:
                            latents = latents.clone().detach().requires_grad_(True)
                            # Forward pass of denoising with text conditioning
                            noise_pred_text = self.unet(latents, t,
                                                        encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                                                        cross_attention_kwargs=cross_attention_kwargs).sample
                            self.unet.zero_grad()
                            e = 0
                            if i in thresholds.keys() and loss_att.item() > 1.0 - thresholds[i]:
                                e += 1
                                del noise_pred_text
                                torch.cuda.empty_cache()
                                
                                loss_att, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                    # TODO
                                    latents=latents,
                                    indices_to_alter=indices_to_alter,
                                    loss=loss_att,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_embeds,
                                    text_input=text_inputs,
                                    attention_store=attention_store,
                                    step_size=scale_factor * np.sqrt(scale_range[i]/10),
                                    t=t,
                                    attention_res=attention_res,
                                    smooth_attentions=smooth_attentions,
                                    sigma=sigma,
                                    kernel_size=kernel_size,
                                    normalize_eot=sd_2_1,
                                    max_refinement_steps=5)

                                # Perform gradient update
                                if i < max_iter_to_alter:
                                    loss_att = self._compute_loss(max_attention_per_index=max_attention_per_index)
                                    if loss_att != 0:
                                        latents = self._update_latent(latents=latents, loss=loss_att,
                                                                    step_size=scale_factor * np.sqrt(scale_range[i]))
                                
                                del loss_att, max_attention_per_index
                                        
                        
                    # TODO perform again
                    latents.detach()
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # TODO masked noise blending
                    noise_source_latents = self.scheduler.add_noise(latents_source, torch.randn_like(latents), t)
                    if mask_image is not None:
                        latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

                    del noise_pred_uncond, noise_pred_text, noise_pred, latent_model_input, noise_source_latents
                    gc.collect()
                    torch.cuda.empty_cache()

            # 8. Post-processing
            # print(latents, latents.grad_fn)
            # print("3", latents.grad_fn)
            # latents = latents.clone().detach().requires_grad_(True)
            # print("4", latents.grad_fn)

        # 9. Run safety checker
        image = self.decode_latents_new(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        del latents
        gc.collect()
        torch.cuda.empty_cache()

        # 10. Convert to PIL
        if output_type == "pil":
            new_image = self.numpy_to_pil(image.detach().cpu().numpy())
            return new_image, image

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
