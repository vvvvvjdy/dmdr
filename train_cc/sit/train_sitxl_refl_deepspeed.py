import logging
import os


import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from pathlib import Path
from collections import OrderedDict
import json

import torch.utils.checkpoint
from tqdm.auto import tqdm

from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.deepspeed import get_active_deepspeed_plugin
from models_sit import SiT_models

from diffusers.models import AutoencoderKL

import math
from torchvision.utils import make_grid
from PIL import Image

from  samplers import v2x0_sampler,pred_v,euler_sampler
from utils import sample_continue,sample_discrete,get_sample,mean_flat
from utils import DINOv2ProcessorWithGrad, ckpt_path_to_accelerator_path
from arguments import parse_args

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



@torch.no_grad()
def copy_main_weights(source, target):
    """
    Copy main weights from source model to target model.
    No lora
    """
    source_main_params = OrderedDict(source.named_parameters())
    target_main_params = OrderedDict(target.named_parameters())

    for name, param in source_main_params.items():
        if "lora" not in name:
            if name in target_main_params:
                target_main_params[name].data.copy_(param.data)
            else:
                print(f"Parameter {name} not found in target model, skipping.")
        else:
            continue





def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ds_config = args.deepspeed_config

    zero2_plugin_a = DeepSpeedPlugin(hf_ds_config=ds_config)
    zero2_plugin_b = DeepSpeedPlugin(hf_ds_config=ds_config)
    deepspeed_plugins = {"z2_a": zero2_plugin_a, "z2_b": zero2_plugin_b}

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins,
    )

    # Since an `AcceleratorState` has already been made, we can just reuse it here
    accelerator2 = Accelerator()

    os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:

        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)

        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}


    gen_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        **block_kwargs
    )

    # load weight for the model
    state_dict_pre = torch.load(args.pretrain_path, map_location='cpu')
    gen_model.load_state_dict(state_dict_pre, strict=False)
    gen_model = gen_model.to(device)
    in_channels = gen_model.in_channels


    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)




    guidance_model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        use_lora  = args.lora_rank > 0,
        lora_rank = args.lora_rank,
        **block_kwargs
    )
    guidance_model.to(device)
    copy_main_weights(gen_model, guidance_model)  # copy weights from the pre-trained model

    all_prams = OrderedDict(guidance_model.named_parameters())
    params_model = []
    # lora finetuning
    if args.lora_rank > 0:
        for name, param in all_prams.items():
            if ("lora" not in name):
                param.requires_grad = False
            elif "lora" in name:
                param.requires_grad = True
                params_model.append(param)

    # full finetuning
    else:
        for name, param in all_prams.items():
                param.requires_grad = True
                params_model.append(param)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    requires_grad(vae, False)  # freeze vae

    # reward_model, we use dinov2_with_head here
    criterion = torch.nn.CrossEntropyLoss()
    if args.encoder_type == "dinov2":
        rep_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc').to(device, dtype=weight_dtype)
        z_size = rep_model.backbone.embed_dim
        if accelerator.is_main_process:
            logger.info(f"Using DINOv2 model with {z_size} feature dimension")

        transform_rep = DINOv2ProcessorWithGrad(res=224)
        rep_model.requires_grad_(False)

    elif args.encoder_type == None:
        rep_model = None
        transform_rep = None
    else:
        raise ValueError(f"Unsupported repmodel: {args.encoder_type}. Supported: dinov2")






    if accelerator.is_main_process:
        logger.info(f"Few-step generator trainable parameters: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad):,}")
        logger.info(f"Guidance units trainable parameters: {sum(p.numel() for p in guidance_model.parameters() if p.requires_grad):,}")



    # Setup optimizer and learning rate scheduler:
    optimizer_gen = torch.optim.AdamW(
        gen_model.parameters(),
        lr=args.learning_rate_gen,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimize those with gradients
    optimizer_groups = []
    if params_model:
        optimizer_groups.append({"params": params_model, "lr": args.learning_rate_gui})


    optimizer_gui = torch.optim.AdamW(
        optimizer_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0,
        eps=args.adam_epsilon,
    )

    local_batch_size = int(args.batch_size)



    if accelerator.is_main_process:
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        log_gui = os.path.join(save_dir, "loss_log", "loss_gui_log.jsonl")
        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gui), exist_ok=True)
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)



    #prepare file to log loss
    if accelerator.is_main_process and args.resume_ckpt is None:
        # clean the log files if they exist
        if os.path.exists(log_gui):
            os.remove(log_gui)
        if os.path.exists(log_gen):
            os.remove(log_gen)
        # add a header to the log files
        with open(log_gui, 'w') as f:
            f.write("loss for guidance_units\n")
        with open(log_gen, 'w') as f:
            f.write("loss for few step generator\n")


    #   create a dummy data loader for accelerator (deepspeed needs it)
    dummy_data = torch.randn(local_batch_size, in_channels, latent_size, latent_size)
    dummy_labels = torch.randint(0, args.num_classes, (local_batch_size,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = torch.utils.data.DataLoader(
        dummy_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    assert get_active_deepspeed_plugin(accelerator.state) is zero2_plugin_a
    gen_model, optimizer_gen, dummy_dataloader = accelerator.prepare(
        gen_model, optimizer_gen, dummy_dataloader
    )

    accelerator2.state.select_deepspeed_plugin("z2_b")
    zero2_plugin_b.deepspeed_config["train_micro_batch_size_per_gpu"] = zero2_plugin_a.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ]

    assert get_active_deepspeed_plugin(accelerator2.state) is zero2_plugin_b
    guidance_model, optimizer_gui = accelerator2.prepare(guidance_model, optimizer_gui)

    # resume:
    global_step = 0
    inner_step = 0
    if args.resume_ckpt is not None:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(
            args.resume_ckpt,
            map_location="cpu",
            weights_only=False
        )
        global_step = ckpt['global_steps']
        inner_step = ckpt['inner_steps']

        acc_dir_1 = ckpt_path_to_accelerator_path(args.resume_ckpt,1)
        acc_dir_2 = ckpt_path_to_accelerator_path(args.resume_ckpt,2)
        accelerator.load_state(acc_dir_1)
        accelerator2.load_state(acc_dir_2)
        if accelerator.is_main_process:
            logger.info(f"Resumed from checkpoint: {args.resume_ckpt} at global step {global_step}")

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 24 // accelerator.num_processes
    yz = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    yz = yz.to(device)
    # Create sampling noise:
    n = yz.size(0)
    xz = torch.randn((n, 4, latent_size, latent_size)).to(device=device,dtype = weight_dtype)


    # For mutistep sampling
    with torch.no_grad():
        with accelerator.autocast():
            samples = euler_sampler(
                gen_model,
                xz,
                yz,
                num_steps=250,
                cfg_scale=args.cfg_r
            )
            samples = vae.decode((samples - latents_bias) / latents_scale).sample
    samples = (samples + 1) / 2.

    # Save images locally
    accelerator.wait_for_everyone()
    out_samples = accelerator.gather(samples.to(torch.float32))

    # Save as grid images
    out_samples = Image.fromarray(array2grid(out_samples))

    if accelerator.is_main_process:
        base_dir = os.path.join(args.output_dir, args.exp_name)
        sample_dir = os.path.join(base_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        out_samples.save(f"{sample_dir}/mutistep_teacher_cfg{args.cfg_r}_500nfe.png")
        logger.info(f"Saved samples for the mutistep teacher model")

    # only for the first steps
    dmd_loss_mean = torch.zeros(1, device=device)
    reward_loss_dino_mean = torch.zeros(1, device=device)


    for inflop in range(inner_step, int(args.max_train_steps * args.ratio_update +100)):  # we add some extra steps to make sure we save the model at the last step
        with accelerator.accumulate([gen_model, guidance_model]):
            gen_model.eval()
            guidance_model.train()
            # samples labels and latents
            latent = torch.randn(local_batch_size, in_channels, latent_size, latent_size).to(device=device,dtype = weight_dtype)
            ys = torch.randint(args.num_classes, size=(local_batch_size,), device=device)

            # backward simulation
            with torch.no_grad():
                with accelerator.autocast():
                    xT, all_x0, t_steps = v2x0_sampler(
                        model=gen_model,
                        latents=latent,
                        y=ys,
                        num_steps=args.num_steps,
                        shift=args.shift,
                    )
            bsz = xT.shape[0]
            dtype = xT.dtype
            # sample timesteps and input latents
            ts = sample_discrete(
                bsz,
                t_steps.flip(0),
                alpha=args.gen_a,
                beta=args.gen_b,
                s_type=args.s_type_gen,
                step=global_step,
                dynamic_step=args.dynamic_step
            ).to(device=accelerator.device, dtype=dtype)
            input_latent_clean = get_sample(
                all_x0,
                ts,
                t_steps
            ).to(device=accelerator.device, dtype=dtype)
            noise_i = torch.randn_like(input_latent_clean)

            input_latent_gen = (1 - ts) * input_latent_clean + ts * noise_i
            v_target = noise_i - input_latent_clean  # may not use
            input_latent_gen = input_latent_gen.to(device=accelerator.device, dtype=dtype)



            # guidance_model update
            ts_gui = sample_continue(
                bsz,
                alpha=args.gui_a,
                beta=args.gui_b,
                s_type=args.s_type_gui,
                step=global_step,
                dynamic_step=args.dynamic_step,
            ).to(device=accelerator.device, dtype=dtype)
            noise = torch.randn_like(input_latent_clean)

            input_latent_gui = (1 - ts_gui) * input_latent_clean + ts_gui * noise
            gt_diff = noise - input_latent_clean
            with accelerator2.autocast():
                v_pred_fake = pred_v(
                    guidance_model,
                    input_latent_gui,
                    ts_gui,
                    ys,
                    cfg_scale=0,
                    lora_scale=args.lora_scale_f
                )
                diffusion_loss = mean_flat((v_pred_fake - gt_diff) ** 2)
                diff_loss_mean = diffusion_loss.mean()
                loss_gui = diff_loss_mean
            accelerator2.backward(loss_gui)
            if accelerator2.sync_gradients:
                params_to_clip = guidance_model.parameters()
                grad_norm = accelerator2.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer_gui.step()
            optimizer_gui.zero_grad(set_to_none=True)
            optimizer_gen.zero_grad(
                set_to_none=True)  # important! This ensures that the generator optimizer does not accumulate gradients

            if (inner_step % args.ratio_update == 0) and inner_step > 0:
                gen_model.train()
                guidance_model.eval()
                # generator update
                with accelerator.autocast():
                    v_pred_gen = pred_v(
                        gen_model,
                        input_latent_gen,
                        ts,
                        ys,
                        cfg_scale=0,
                        lora_scale=0,
                    )
                x0 = input_latent_gen + (0.0 - ts) * v_pred_gen

                # reward loss
                if global_step >= args.cold_start_iter:
                    if args.encoder_type is not None:
                        with accelerator.autocast():
                            # convert vae latent to dino input

                            latent_x0 = (x0 - latents_bias) / latents_scale
                            samples = vae.decode(latent_x0).sample
                            samples = ((samples + 1) / 2.).clamp(0, 1)
                            samples = transform_rep(samples)
                            if args.encoder_type == 'dinov2':
                                logistics = rep_model(samples)

                                dino_loss = criterion(logistics, ys)
                                reward_loss_dino_mean = dino_loss.mean()
                            else:
                                raise NotImplementedError(f"Encoder type {args.encoder_type} not implemented.")


                # dmd loss
                if global_step < args.dynamic_step:
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * global_step / args.dynamic_step))
                    lora_scale_r = args.lora_scale_r * cosine_factor
                else:
                    lora_scale_r = 0.0
                ts_dmd = sample_continue(
                    bsz,
                    alpha=args.gui_a,
                    beta=args.gui_b,
                    s_type=args.s_type_gui,
                    step=global_step,
                    dynamic_step=args.dynamic_step,
                ).to(device=accelerator.device, dtype=dtype)
                noise = torch.randn_like(input_latent_gen)
                input_latent_dmd = (1 - ts_dmd) * x0 + ts_dmd * noise
                with accelerator.autocast():
                    with torch.no_grad():
                        v_pred_fake = pred_v(
                            guidance_model,
                            input_latent_dmd,
                            ts_dmd,
                            ys,
                            cfg_scale=0,
                            lora_scale=args.lora_scale_f
                        )
                        v_pred_real = pred_v(
                            guidance_model,
                            input_latent_dmd,
                            ts_dmd,
                            ys,
                            cfg_scale= args.cfg_r,
                            lora_scale=lora_scale_r
                        )

                    x0_r = input_latent_dmd + (0.0 - ts_dmd) * v_pred_real
                    x0_f = input_latent_dmd + (0.0 - ts_dmd) * v_pred_fake
                    p_real = x0 - x0_r
                    p_fake = x0 - x0_f

                    # following the DMD and SDS, we use the estimation of the grad direction to optimize the generator
                    grad = (p_real - p_fake) / (
                                torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                    grad = torch.nan_to_num(grad)
                    dmd_loss = 0.5 * torch.nn.functional.mse_loss(x0.float(),
                                                                  (x0 - grad).detach().float(),
                                                                  reduction='mean')
                    dmd_loss_mean = dmd_loss.mean()

                    # gathering all losses for backward
                    loss_gen = dmd_loss_mean + args.dino_loss_weight * reward_loss_dino_mean
                accelerator.backward(loss_gen)
                if accelerator.sync_gradients:
                    params_to_clip = gen_model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_gen.step()
                optimizer_gen.zero_grad(set_to_none=True)
                optimizer_gui.zero_grad(set_to_none=True)

        logs = {
            "diff": diff_loss_mean.item(),
            "dmd": dmd_loss_mean.item(),
            "dino": reward_loss_dino_mean.item(),
            "in_s": inner_step,
            "glo_s": global_step,
        }
        if accelerator.sync_gradients:
            if (inner_step % args.ratio_update == 0) and inner_step > 0:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    with open(log_gen, 'a') as f:
                        json.dump(logs, f)
                        f.write("\n")
                accelerator.log(logs, step=global_step)
            inner_step += 1
            if accelerator.is_main_process:
                with open(log_gui, 'a') as f:
                    json.dump(logs, f)
                    f.write("\n")
        progress_bar.set_postfix(**logs)

        if accelerator.sync_gradients:
            # save checkpoint (feel free to adjust the frequency)
            if (global_step % args.checkpoint_steps == 0) and (
                    inner_step % args.ratio_update == 0) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint = {
                        "global_steps": global_step,
                        "inner_steps": inner_step,
                    }
                    checkpoint_dir_torch = os.path.join(checkpoint_dir, f"torch_units")
                    os.makedirs(checkpoint_dir_torch, exist_ok=True)
                    checkpoint_path = f"{checkpoint_dir_torch}/step-{global_step}.pt"
                    torch.save(checkpoint, checkpoint_path)

                # save accelerator states
                acc1_dir = os.path.join(checkpoint_dir, "accelerator_1")
                acc2_dir = os.path.join(checkpoint_dir, "accelerator_2")
                accelerator_state_dir = os.path.join(acc1_dir, f"step-{global_step}")
                accelerator2_state_dir = os.path.join(acc2_dir, f"step-{global_step}")
                if accelerator.is_main_process:
                    os.makedirs(acc1_dir, exist_ok=True)
                    os.makedirs(acc2_dir, exist_ok=True)
                    os.makedirs(accelerator_state_dir, exist_ok=True)
                    os.makedirs(accelerator2_state_dir, exist_ok=True)

                accelerator.wait_for_everyone()
                accelerator.save_state(accelerator_state_dir)
                accelerator.wait_for_everyone()
                accelerator2.save_state(accelerator2_state_dir)

                if accelerator.is_main_process:
                    logger.info(f"Saved checkpoint")

            # sample and save images (feel free to adjust the frequency)
            if ((global_step % args.sample_steps == 0) and (
                    inner_step % args.ratio_update == 0) and global_step > 0) or inner_step == 1:

                with torch.no_grad():
                    with accelerator.autocast():
                        samples = v2x0_sampler(
                            model=gen_model,
                            latents=xz,
                            y=yz,
                            num_steps=args.num_steps,
                            shift=args.shift,
                        )[0]
                        samples = vae.decode((samples - latents_bias) / latents_scale).sample
                samples = (samples + 1) / 2.

                # Save images locally
                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(samples.to(torch.float32))

                # Save as grid images
                out_samples = Image.fromarray(array2grid(out_samples))

                if accelerator.is_main_process:
                    base_dir = os.path.join(args.output_dir, args.exp_name)
                    sample_dir = os.path.join(base_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                    logger.info(f"Saved samples at step {global_step}")



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training Done!")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)










        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
