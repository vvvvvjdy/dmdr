from typing import Any, Dict, List, Optional, Union
import torch
import math

def pred_v(
        transformer,
        noisy_latents,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        timesteps,
        guidance_scale=1.0,
        joint_attention_kwargs=None,
):
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        timesteps = timesteps.repeat(2)
    v_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=False,
    )[0]
    if do_cfg:
        v_pred_uncond, v_pred_cond = v_pred.chunk(2)
        v_pred = v_pred_uncond + guidance_scale * (v_pred_cond - v_pred_uncond)
    return v_pred.to(noisy_latents.dtype)



# In dmd, we always hypo that the model predicting x0, so we need to modify the sampling process accordingly.
# Here, we do not use cfg as other concurrent works, but we still make the sampler like those implemented in diffusers.
def v2x0_sampler(
        pipe,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_layer_guidance_scale: float = 2.8,
        shift: float = 1.0,
        sample_type = 'sde',
        return_imgs=False,
        special_transformer = None,
):
    
    # Checker and Define
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor


    pipe.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipe._guidance_scale = guidance_scale
    pipe._skip_layer_guidance_scale = skip_layer_guidance_scale
    pipe._clip_skip = clip_skip
    pipe._joint_attention_kwargs = joint_attention_kwargs
    pipe._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipe._execution_device

    lora_scale = (
        pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=pipe.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    # Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
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
    t_steps = torch.linspace(1.0, 0.001, num_inference_steps + 1, dtype=latents.dtype)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0.001

    # Prepare_emb
    all_x0 = []
    x_next = latents

    # Denoising loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = (t_cur.expand(latents.shape[0]) * 1000).round().to(device=device, dtype=latents.dtype)
        if special_transformer is not None:
            v_pred = special_transformer(
                hidden_states=x_cur,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]
        else:
            v_pred = pipe.transformer(
                hidden_states=x_cur,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]
        v_pred = v_pred.to(latents.dtype)

        if sample_type == 'sde':
            x_0 = x_cur + (t_steps[-1] - t_cur) * v_pred
            all_x0.append(x_0.to(latents.dtype))
            noise_add = torch.randn_like(x_0)
            x_next = (1 - t_next) * x_0 + t_next * noise_add

        elif sample_type == 'ode':
            x_next = x_cur + (t_next - t_cur) * v_pred
            x_0 = x_next

        else:
            raise NotImplementedError(f"sample_type {sample_type} not implemented.")

    # we use a fix x0 state for ode sampling
    if sample_type == 'ode':
        for ij in range(len(t_steps) - 1):
            all_x0.append(x_0.to(latents.dtype))

    latents = x_0
    if return_imgs:
        # Decode the latents to images
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        latents = latents.to(dtype=pipe.vae.dtype)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type=output_type)
        return image, all_x0, t_steps[:-1]
    else:
        return 0, all_x0, t_steps[:-1]  # Return dummy value for image if not needed


