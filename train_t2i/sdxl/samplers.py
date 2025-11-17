from typing import Any, Dict, List, Optional, Union
import torch

def get_x0_from_noise(sample, model_output, timestep, alphas_cumprod):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    # 0.0047 corresponds to the alphas_cumprod of the last timestep (999)
    # alpha_prod_t = (torch.ones_like(timestep) * 0.0047).reshape(-1, 1, 1, 1).double()
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample



def build_condition_time( resolution):
    original_size = (resolution, resolution)
    target_size = (resolution, resolution)
    crop_top_left = (0, 0)

    add_time_ids = list(original_size + crop_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    return add_time_ids

def pred_noise(
        unet,
        noisy_latents,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        timesteps=None,
        guidance_scale=1.0,
        cross_attention_kwargs=None,
        res= 1024,
):
    bsz = noisy_latents.shape[0]
    do_cfg = guidance_scale > 1.0
    add_time_ids = build_condition_time(res).repeat(bsz, 1).to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        timesteps = torch.cat([timesteps, timesteps], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False,
    )[0]
    if do_cfg:
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred.to(noisy_latents.dtype)



# In dmd, we always hypo that the model predicting x0, so we need to modify the sampling process accordingly.
# Here, we do not use cfg as other concurrent works, but we still make the sampler like those implemented in diffusers.
def noise2x0_sampler(
        pipe,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        skip_layer_guidance_scale: float = 2.8,
        shift: float = 1.0,
        return_imgs=False,
        noise_scheduler=None,
        special_unet=None,
):
    # Checker and Define
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    device = pipe._execution_device

    # Set noise scheduler
    orginal_scheduler = pipe.scheduler
    pipe.scheduler = noise_scheduler if noise_scheduler is not None else orginal_scheduler
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)


    pipe._guidance_scale = guidance_scale
    pipe._skip_layer_guidance_scale = skip_layer_guidance_scale
    pipe._clip_skip = clip_skip
    pipe._cross_attention_kwargs = cross_attention_kwargs
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    lora_scale = (
        pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        lora_scale=lora_scale,
    )

    # Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )
    # Prepare timesteps
    t_steps = torch.linspace(999, 0, num_inference_steps + 1, dtype=latents.dtype)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0
    # for sdxl, t_steps should be in [0,1000) Integer
    t_steps = (t_steps.round()).to(device)

    # we follow dmd to set conditional timestep as 399 for 1 step generation
    # This is because for SDXL, set the timestep to 999 will lead to artifacts
    if num_inference_steps == 1:
        t_steps[0] = 399

    # Prepare extra kwargs. cond

    add_time_ids = build_condition_time((height + width) // 2).repeat(batch_size, 1).to(device=latents.device,
                                                                                 dtype=latents.dtype)

    # Prepare_emb
    all_x0 = []
    x_next = latents
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

    # Denoising loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t_cur.expand(batch_size).to(device=device, dtype=torch.long)

        if special_unet is not None:
            noise_pred = special_unet(
                x_cur,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                return_dict=False,
            )[0]
        else:
            noise_pred = pipe.unet(
                x_cur,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=pipe.cross_attention_kwargs,
                return_dict=False,
            )[0]
        noise_pred = noise_pred.to(latents.dtype)
        x_0 = get_x0_from_noise(x_cur, noise_pred, timestep, alphas_cumprod)
        all_x0.append(x_0.to(latents.dtype))
        next_timestep = t_next.expand(batch_size).to(device=device, dtype=torch.long)
        x_next = noise_scheduler.add_noise(x_0, torch.randn_like(latents), next_timestep).to(latents.dtype)

    latents = all_x0[-1]

    # restore scheduler
    pipe.scheduler = orginal_scheduler

    if return_imgs:
        # Decode the latents to images
        latents = (latents / pipe.vae.config.scaling_factor)
        latents = latents.to(dtype=pipe.vae.dtype)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        return image, all_x0, t_steps[:-1]
    else:
        return 0, all_x0, t_steps[:-1]  # Return dummy value for image if not needed


