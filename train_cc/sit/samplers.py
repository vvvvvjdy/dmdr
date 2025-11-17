import torch
import numpy as np


def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",  # not used, just for compatability
):
    # setup conditioning
    _dtype = latents.dtype 
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=_dtype)
    x_next = latents.to(_dtype)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=_dtype) * t_cur
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            ).to(_dtype)
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=_dtype
                ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                ).to(_dtype)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next



def pred_v(
    model,
    latents,
    t,
    y,
    cfg_scale,
    lora_scale=0,
    num_of_calsses=1000,

):
    do_cfg = cfg_scale > 1.0
    if do_cfg:
        latents = torch.cat([latents, latents], dim=0)
        t = torch.cat([t, t], dim=0)
        y_null = torch.tensor([num_of_calsses] * y.size(0), device=y.device)
        y = torch.cat([y_null, y], dim=0)
    time_input = t.flatten()

    v_pred = model(
        latents,
        time_input,
        y,
        lora_scale=lora_scale,
    )
    if do_cfg:
        v_pred_uncond, v_pred_cond = v_pred.chunk(2)
        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
    return v_pred.to(latents.dtype)



# In dmd, we always hypo that the model predicting x0, so we need to modify the sampling process accordingly.
# We only implement sde sampling here.
def v2x0_sampler(
        model,
        latents,
        y,
        num_steps=20,
        shift = 1,

):
    # Prepare timesteps
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=latents.dtype)
    t_steps = shift * t_steps / (1 + (shift - 1) * t_steps)
    t_steps[-1] = 0.0

    x_next = latents
    device = x_next.device
    all_x0 = []
    _dtype = latents.dtype

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            model_input = x_cur
            y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=_dtype) * t_cur
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            )

            # get xt to x0
            x_0 = x_cur + (0 - t_cur) * d_cur
            all_x0.append(x_0)
            x_next = (1 - t_next) * x_0 + t_next * torch.randn_like(x_0)
    return x_next, all_x0, t_steps[:-1]
