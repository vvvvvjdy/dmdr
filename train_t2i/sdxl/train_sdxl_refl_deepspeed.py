import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import torch
from huggingface_hub import hf_hub_download
from accelerate import Accelerator, DeepSpeedPlugin, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils.deepspeed import get_active_deepspeed_plugin
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline,LCMScheduler,DDIMScheduler,AutoencoderTiny
from transformers import AutoTokenizer, AutoModel
from utils import get_sample,mean_flat,encode_prompt,create_generator,sample_continue,sample_discrete,ckpt_path_to_accelerator_path
from diffusers.utils.torch_utils import is_compiled_module
from open_clip import create_model_and_transforms
import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
import logging
from pathlib import Path
import json
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from samplers import noise2x0_sampler,pred_noise,get_x0_from_noise
import copy
import math
from torchvision.utils import make_grid
from PIL import Image
from dataset import TextPromptDataset
from rl_utils import CLIPImageProcessorTensor, clip_score
from arguments import  parse_args

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


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

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
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


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16


    # Create pipe :
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
    )

    if 'stable-diffusion-xl' in args.pretrained_model:
        # use the tiny vae for sdxl
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float32,)
        pipeline.vae = vae
        if accelerator.is_main_process:
            logger.info(f"Using tiny VAE for SDXL to support mix percision training")



    noise_scheduler_gen = LCMScheduler.from_config(pipeline.scheduler.config) # we use LCM scheduler for few step generation
    alphas_cumprod_gen = noise_scheduler_gen.alphas_cumprod.to(device = device, dtype=inference_dtype)

    noise_scheduler_gui = DDIMScheduler.from_config(pipeline.scheduler.config) # we use DDIM scheduler for guidance units
    alphas_cumprod_gui = noise_scheduler_gui.alphas_cumprod.to(device = device, dtype=inference_dtype)


    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(args.use_lora_gen <= 1)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    # disable safety checker
    pipeline.safety_checker = None

    #create reward model
    if args.clip_reward_model is not None:
        crop_size = 224
        clip_image_processor = CLIPImageProcessorTensor(crop_size=crop_size,model_type="clip")
        clip_text_processor = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        clip_model,_,_ = create_model_and_transforms(
        'ViT-H-14',
        args.clip_reward_model,
        precision=inference_dtype,
        device=torch.device('cpu'),
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )
         # freeze the clip model
        clip_model.requires_grad_(False)
        clip_model.eval()

    if args.hpsv2_reward_model is not None:
        crop_size = 224
        hps2_image_processor = CLIPImageProcessorTensor(crop_size=crop_size,model_type="clip")
        hps2_text_processor = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        hps2_model, _,_ = create_model_and_transforms(
            'ViT-H-14',
            precision=inference_dtype,
            device=torch.device('cpu'),
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
        )

        ckpt_path = args.hpsv2_reward_model
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        hps2_model.load_state_dict(checkpoint['state_dict'], strict=True)
        if accelerator.is_main_process:
            logger.info(f"Loaded HPSv2 model from {ckpt_path} successfully")
        hps2_model.requires_grad_(False)
        hps2_model.eval()



    if args.cold_start_iter < 0:
        # make the progress bar nicer
        pipeline.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )
    else:
        # disable progress bar for cold start
        pipeline.set_progress_bar_config(disable=True)

    # Use deepcopy to avoid modifying the original model
    guidance_model = copy.deepcopy(pipeline.unet)

    # Load ode-pretrain weight for 1-step sdxl-model
    # This is important for one step sdxl to mitigate artifacts
    # Meanwhile, we do not use dyna-c2f for one step sdxl because we find shift on higher noise could also cause artifacts
    # Note that this is only for one step sdxl, suggesting this issue might be specific to this model.

    if args.sample_num_steps==1:
        args.s_type_gui == "uniform"
        if args.xl_ode_pretrain_path is None:
            repo_name = "tianweiy/DMD2"
            ckpt_name = "model/sdxl/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399_checkpoint_model_002000.bin"
            pipeline.unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cpu"))

        else:
            pipeline.unet.load_state_dict(torch.load(args.xl_ode_pretrain_path, map_location="cpu"))
        if accelerator.is_main_process:
            logger.info(f"Loaded 1-step sd-xl model from {args.xl_ode_pretrain_path} successfully")

    # init lora
    if args.use_lora > 1:
        # Set correct lora layers
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ]
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        guidance_model = get_peft_model(guidance_model, unet_lora_config)

        if args.use_lora_gen > 1:
            unet_lora_config_gen = LoraConfig(
                r=args.lora_rank * 2,
                lora_alpha=args.lora_rank * 4,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            pipeline.unet = get_peft_model(pipeline.unet, unet_lora_config_gen)




    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)

    if args.clip_reward_model is not None:
        clip_model.to(accelerator.device, dtype=inference_dtype)
    if args.hpsv2_reward_model is not None:
        hps2_model.to(accelerator.device, dtype=inference_dtype)



    gen_model = pipeline.unet
    guidance_model_trainable_parameters = list(filter(lambda p: p.requires_grad, guidance_model.parameters()))
    gen_model_trainable_parameters = list(filter(lambda p: p.requires_grad, gen_model.parameters()))

    # Prepare the lora scale for fake and real estimation
    scale_dict_r = {"scale": args.lora_scale_r}
    scale_dict_f = {"scale": args.lora_scale_f}


    # In dmd, we find EMA is not useful, so we do not preserve an EMA model
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup optimizer and learning rate scheduler:
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer_gen = optimizer_cls(
        gen_model_trainable_parameters,
        lr=args.learning_rate_gen,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_gui = optimizer_cls(
        guidance_model_trainable_parameters,
        lr=args.learning_rate_gui,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0,
        eps=args.adam_epsilon,
    )
    # Setup dataset:
    train_dataset = TextPromptDataset(args.data_path_train)
    test_dataset = TextPromptDataset()

    num_prompts = len(train_dataset)
    local_batch_size = int(args.batch_size)

    # Create data loaders:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    #neg prompts
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers,
                                                                        max_sequence_length=128,
                                                                        device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(local_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(local_batch_size, 1)


    # printing
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_prompts} prompts")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"Total trainable parameters in gen_model: {sum(p.numel() for p in gen_model.parameters() if p.requires_grad)}")
        logger.info(f"Total trainable parameters in guidance_model: {sum(p.numel() for p in guidance_model.parameters() if p.requires_grad)}")

        log_gui = os.path.join(save_dir, "loss_log", "loss_gui_log.jsonl")
        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gui), exist_ok=True)
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)


    # prepare file to log loss
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


    # Prepare models for training:
    assert get_active_deepspeed_plugin(accelerator.state) is zero2_plugin_a
    gen_model, optimizer_gen, train_dataloader, test_dataloader = accelerator.prepare(
        gen_model, optimizer_gen, train_dataloader, test_dataloader
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
    epoch_start = -1
    if args.resume_ckpt is not None:
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(
            args.resume_ckpt,
            map_location="cpu",
            weights_only=False
        )
        epoch_start = ckpt['epoch'] - 1
        global_step = ckpt['global_steps']
        inner_step = ckpt['inner_steps']

        acc_dir_1 = ckpt_path_to_accelerator_path(args.resume_ckpt, 1)
        acc_dir_2 = ckpt_path_to_accelerator_path(args.resume_ckpt, 2)
        accelerator.load_state(acc_dir_1)
        accelerator2.load_state(acc_dir_2)
        if accelerator.is_main_process:
            logger.info(
                f"Resumed from checkpoint: {args.resume_ckpt} at epoch {epoch_start}, global step {global_step}")

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    ############################################### Train Loop ######################################################

    # get sample prompts, free to change
    test_prompts, _ = next(iter(test_dataloader))


    with torch.no_grad():
        generator_test = create_generator(test_prompts,2026)
        # get embeddings
        prompt_embeds_test, pooled_prompt_embeds_test = compute_text_embeddings(
            test_prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )
        # sample multistep images for comparison
        if global_step == 0:
            with accelerator.autocast():
                with pipeline.unet.disable_adapter() if args.use_lora_gen > 1 else torch.no_grad():
                    images = pipeline(
                        prompt_embeds=prompt_embeds_test,
                        pooled_prompt_embeds=pooled_prompt_embeds_test,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: prompt_embeds_test.shape[0]],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[
                            : pooled_prompt_embeds_test.shape[0]],
                        num_inference_steps=25,
                        output_type="pt",
                        height=1024,
                        width=1024,
                        guidance_scale=args.cfg_r,
                        generator=generator_test,
                    )[0]


            #resize to 512 (2,3, resolution, resolution) to (2,3,512,512)
            images = torch.nn.functional.interpolate(images, size=(512, 512), mode='bicubic', align_corners=False)


            # Save images locally
            accelerator.wait_for_everyone()
            out_samples = accelerator.gather(images.to(torch.float32))

            # Save as grid images
            out_samples = Image.fromarray(array2grid(out_samples))
            if accelerator.is_main_process:
                base_dir = os.path.join(args.output_dir, args.exp_name)
                sample_dir = os.path.join(base_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                out_samples.save(f"{sample_dir}/samples_multistep_cfg{args.cfg_r}.png")
                logger.info(f"Saved multistep samples")

    # only for the first steps
    dmd_loss_mean = torch.zeros(1, device=accelerator.device)
    reward_loss_clip_mean = torch.zeros(1, device=accelerator.device)
    reward_loss_hps_mean = torch.zeros(1, device=accelerator.device)
    reward_loss_clip_hinge = torch.zeros(1, device=accelerator.device)
    reward_loss_hps_hinge = torch.zeros(1, device=accelerator.device)
    for epoch in range(epoch_start + 1, args.epochs):
        for prompts, metap in train_dataloader:

            with accelerator.accumulate([gen_model, guidance_model]):
                gen_model.eval()
                guidance_model.train()

                with torch.no_grad():
                    # get embeddings
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                        prompts,
                        text_encoders,
                        tokenizers,
                        max_sequence_length=128,
                        device=accelerator.device
                    )


                    # backward simulation
                    if True:
                        with accelerator.autocast():
                            _, all_x0,t_steps = noise2x0_sampler(
                                pipeline,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_prompt_embeds=sample_neg_prompt_embeds,
                                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                                num_inference_steps=args.sample_num_steps,
                                output_type="pt",
                                height=args.resolution,
                                width=args.resolution,
                                generator=None,
                                shift = args.shift,
                                noise_scheduler=noise_scheduler_gen,
                                return_imgs=False,
                            )
                        bsz = all_x0[0].shape[0]
                        dtype = all_x0[0].dtype

                        # sample timesteps and input latents
                        ts = sample_discrete(
                            bsz,
                            t_steps.flip(0),
                            alpha=args.gen_a,
                            beta=args.gen_b,
                            s_type = args.s_type_gen,
                            step=global_step,
                            dynamic_step=args.dynamic_step
                        ).to(device=accelerator.device, dtype= dtype)
                        input_latent_clean = get_sample(
                            all_x0,
                            ts,
                            t_steps
                        ).to(device=accelerator.device, dtype=dtype)
                        noise_i = torch.randn_like(input_latent_clean)

                        ts = ts.to(device = accelerator.device, dtype=torch.long)

                        input_latent_gen = noise_scheduler_gen.add_noise(input_latent_clean, noise_i, ts)
                        input_latent_gen = input_latent_gen.to(device=accelerator.device, dtype=dtype)
                        # convert last step noise to gaussian noise, important!
                        if args.sample_num_steps >1:
                            input_latent_gen[ts == 999] = torch.randn_like(input_latent_gen[ts == 999])
                        else:
                            input_latent_gen = torch.randn_like(input_latent_gen)



                # guidance_model update
                ts_gui = sample_continue(
                    bsz,
                    alpha=4,
                    beta=1.5,
                    s_type=args.s_type_gui,
                    step=global_step,
                    dynamic_step=args.dynamic_step,
                ).to(device=accelerator.device, dtype=dtype)
                ts_gui = (ts_gui * 1000).round().clamp(min=0, max=999)
                ts_gui = ts_gui.to( device = accelerator.device, dtype=torch.long)
                noise = torch.randn_like(input_latent_clean)
                input_latent_gui = noise_scheduler_gui.add_noise(input_latent_clean, noise, ts_gui)
                gt_diff = noise


                with accelerator2.autocast():
                    noise_pred_fake = pred_noise(
                        unet = guidance_model,
                        noisy_latents=input_latent_gui,
                        timesteps=ts_gui,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=0.0,
                        cross_attention_kwargs=scale_dict_f,
                        res=args.resolution,
                    )
                    diffusion_loss = mean_flat((noise_pred_fake - gt_diff) ** 2)
                    diff_loss_mean = diffusion_loss.mean()
                    loss_gui = diff_loss_mean
                accelerator2.backward(loss_gui)
                if accelerator2.sync_gradients:
                    params_to_clip =  guidance_model.parameters()
                    grad_norm = accelerator2.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_gui.step()
                optimizer_gui.zero_grad(set_to_none=True)
                optimizer_gen.zero_grad(set_to_none=True)  # important! This ensures that the generator optimizer does not accumulate gradients

                if (inner_step % args.ratio_update == 0) and inner_step > 0:
                    gen_model.train()
                    guidance_model.eval()
                    # generator update
                    with accelerator.autocast():
                        noise_pred_gen = pred_noise(
                            unet=gen_model,
                            noisy_latents=input_latent_gen,
                            timesteps=ts,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            guidance_scale=0.0,
                            cross_attention_kwargs=pipeline.cross_attention_kwargs,
                            res=args.resolution,
                        )


                    x0 = get_x0_from_noise(input_latent_gen, noise_pred_gen, ts, alphas_cumprod_gen)



                    # reward loss
                    if global_step >= args.cold_start_iter:
                        if args.clip_reward_model is not None or args.hpsv2_reward_model is not None:
                            with accelerator.autocast():
                                reward_input = (x0 / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                                reward_input = pipeline.vae.decode(reward_input, return_dict=False)[0]
                                reward_input = pipeline.image_processor.postprocess(reward_input, output_type="pt")

                        if args.clip_reward_model is not None:
                            with accelerator.autocast():
                                clip_img_feat = clip_image_processor(reward_input).to(device=accelerator.device, dtype=inference_dtype)
                                clip_text_feat = clip_text_processor(
                                    prompts,
                                    padding="max_length",
                                    max_length=77,
                                    truncation=True,
                                    return_tensors="pt",
                                ).to(device=accelerator.device)
                                clip_img_feat = clip_model.encode_image(clip_img_feat["pixel_values"])
                                clip_text_feat = clip_model.encode_text(clip_text_feat.input_ids)
                                reward_loss_clip= clip_score(
                                    clip_img_feat,
                                    clip_text_feat,
                                    1,
                                    0
                                )
                                reward_loss_clip_hinge = torch.nn.functional.relu(-reward_loss_clip + 0.7).mean()
                                reward_loss_clip_mean = reward_loss_clip.mean()

                        if args.hpsv2_reward_model is not None:
                            with accelerator.autocast():
                                hps2_img_feat = hps2_image_processor(reward_input).to(device=accelerator.device, dtype=inference_dtype)
                                if args.magic_prompt is not None:
                                    prompts_hps = [args.magic_prompt + " " + prompts[kj] for kj in range(len(prompts))]
                                else:
                                    prompts_hps = prompts
                                hps2_text_feat = hps2_text_processor(
                                    prompts_hps,
                                    padding="max_length",
                                    max_length=77,
                                    truncation=True,
                                    return_tensors="pt",
                                ).to(device=accelerator.device)
                                hps2_outputs = hps2_model(image=hps2_img_feat["pixel_values"], text=hps2_text_feat.input_ids)
                                hps2_img_feat, hps2_text_fit = hps2_outputs["image_features"], hps2_outputs["text_features"]
                                logits_per_image_hps2 = hps2_img_feat @ hps2_text_fit.T
                                reward_loss_hps = torch.diagonal(logits_per_image_hps2)
                                reward_loss_hps_hinge = torch.nn.functional.relu(-reward_loss_hps + 0.7).mean()
                                reward_loss_hps_mean = reward_loss_hps.mean()


                    # dmd loss
                    if global_step < args.dynamic_step:
                        cosine_factor = 0.5 * (1 + math.cos(math.pi * global_step / args.dynamic_step))
                        lora_scale_r = args.lora_scale_r * cosine_factor
                    else:
                        lora_scale_r = 0.0
                    scale_dict_r = {"scale": lora_scale_r}
                    ts_dmd = sample_continue(
                    bsz,
                    alpha=args.gui_a,
                    beta=args.gui_b,
                    s_type= args.s_type_gui,
                    step=global_step,
                    dynamic_step=args.dynamic_step,
                ).to(device=accelerator.device, dtype=dtype)
                    noise = torch.randn_like(input_latent_clean)
                    ts_dmd = (ts_dmd * 1000).round().clamp(min=0, max=999)
                    ts_dmd = ts_dmd.to(device = accelerator.device, dtype=torch.long)
                    input_latent_dmd = noise_scheduler_gui.add_noise(x0, noise, ts_dmd)



                    with accelerator.autocast():
                        with torch.no_grad():

                            noise_pred_fake = pred_noise(
                                unet=guidance_model,
                                noisy_latents=input_latent_dmd,
                                timesteps=ts_dmd,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_prompt_embeds=sample_neg_prompt_embeds,
                                negative_pooled_prompt_embeds= sample_neg_pooled_prompt_embeds,
                                guidance_scale= random.uniform(args.cfg_f - 0.5, args.cfg_f + 0.5) if args.cfg_f > 1 else 1.0,
                                cross_attention_kwargs=scale_dict_f,
                                res = args.resolution,
                            )

                            noise_pred_real = pred_noise(
                                unet=guidance_model,
                                noisy_latents=input_latent_dmd,
                                timesteps=ts_dmd,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_prompt_embeds=sample_neg_prompt_embeds,
                                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                                guidance_scale=random.uniform(args.cfg_r - 0.5, args.cfg_r + 0.5) if args.cfg_r > 1 else 1.0,
                                cross_attention_kwargs=scale_dict_r,
                                res=args.resolution,
                            )
                        x0_r = get_x0_from_noise(input_latent_dmd, noise_pred_real, ts_dmd, alphas_cumprod_gui)
                        x0_f = get_x0_from_noise(input_latent_dmd, noise_pred_fake, ts_dmd, alphas_cumprod_gui)
                        p_real = x0 - x0_r
                        p_fake = x0 - x0_f

                        # following the DMD and SDS, we use the estimation of the grad direction to optimize the generator
                        grad = (p_real - p_fake) / (torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
                        grad = torch.nan_to_num(grad)
                        dmd_loss = 0.5 * torch.nn.functional.mse_loss(x0.float(), (x0 - grad).detach().float(),
                                                                      reduction='mean')
                        dmd_loss_mean =  dmd_loss.mean()


                        # gathering all losses for backward
                        loss_gen = dmd_loss_mean + reward_loss_clip_hinge * args.reward_clip_weight + reward_loss_hps_hinge * args.reward_hps_weight
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
                "clip": reward_loss_clip_mean.item(),
                "hps": reward_loss_hps_mean.item(),
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
                        inner_step % args.ratio_update == 0) and global_step > 500:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint = {
                            "epoch": epoch,
                            "global_steps": global_step,
                            "inner_steps": inner_step,
                        }
                        checkpoint_dir_torch = os.path.join(checkpoint_dir, f"torch_units")
                        os.makedirs(checkpoint_dir_torch, exist_ok=True)
                        checkpoint_path = f"{checkpoint_dir_torch}/step-{global_step}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        lora_dict_gui_base = os.path.join(checkpoint_dir, f"lora_gui")
                        os.makedirs(lora_dict_gui_base, exist_ok=True)
                        lora_dict_gui = os.path.join(lora_dict_gui_base, f"step-{global_step}")
                        os.makedirs(lora_dict_gui, exist_ok=True)
                        unwrap_model(guidance_model, accelerator).save_pretrained(lora_dict_gui)
                        if args.use_lora_gen > 1:
                            lora_dict_gen_base = os.path.join(checkpoint_dir, f"lora_gen")
                            os.makedirs(lora_dict_gen_base, exist_ok=True)
                            lora_dict_gen = os.path.join(lora_dict_gen_base, f"step-{global_step}")
                            os.makedirs(lora_dict_gen, exist_ok=True)
                            unwrap_model(gen_model, accelerator).save_pretrained(lora_dict_gen)

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
                            images = noise2x0_sampler(
                                pipeline,
                                prompt_embeds=prompt_embeds_test,
                                pooled_prompt_embeds=pooled_prompt_embeds_test,
                                negative_prompt_embeds=sample_neg_prompt_embeds[: prompt_embeds_test.shape[0]],
                                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[
                                    : pooled_prompt_embeds_test.shape[0]],
                                num_inference_steps=args.sample_num_steps,
                                output_type="pt",
                                height=args.resolution,
                                width=args.resolution,
                                generator=generator_test,
                                shift=args.shift,
                                noise_scheduler=noise_scheduler_gen,
                                return_imgs=True,
                            )[0]
                        # resize to 512 (2,3, resolution, resolution) to (2,3,512,512)
                        images = torch.nn.functional.interpolate(images, size=(512, 512), mode='bicubic',
                                                                 align_corners=False)


                    # Save images locally
                    accelerator.wait_for_everyone()
                    out_samples = accelerator.gather(images.to(torch.float32))

                    # Save as grid images
                    out_samples = Image.fromarray(array2grid(out_samples))
                    if accelerator.is_main_process:
                        base_dir = os.path.join(args.output_dir, args.exp_name)
                        sample_dir = os.path.join(base_dir, "samples")
                        os.makedirs(sample_dir, exist_ok=True)
                        out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                        logger.info(f"Saved samples at step {global_step}")

            if global_step > args.max_train_steps:
                break
        if global_step > args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training Done!")
    accelerator.end_training()




if __name__ == "__main__":
    args = parse_args()
    main(args)





























