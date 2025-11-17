import torch
import hashlib
import os
def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2 ** 31)  # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def sample_discrete(
        B: int,
        timesteps: list,
        alpha: float = 4.0,
        beta: float = 1.0,
        s_type: str = "logit_normal",
        step: int = 0,
        dynamic_step: int = 1000,

) -> torch.Tensor:
    """
    B: batch size
    timesteps: list of discrete values to sample from
    m: mean of the logit normal distribution
    s: standard deviation of the logit normal distribution
    s_type: type of sampling, choose from ["logit_normal","uniform"]
    """
    if s_type == "uniform":
        indices = torch.randint(0, len(timesteps), (B,))
        discrete_samples = torch.tensor([timesteps[i] for i in indices])
        return discrete_samples.reshape(B,1,1,1) # Reshape to [B, 1, 1, 1] for broadcasting
    elif s_type == "logit_normal":
        # Logit-normal sampling
        if dynamic_step > 0:
            progress = min(step / dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141593)))
            alpha = 1.0 + (alpha - 1.0) * cosine_decay
            beta = 1.0 + (beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(alpha, beta).sample((B,))
        timesteps_tensor = timesteps.reshape(1, -1)
        distances = torch.abs(t.unsqueeze(-1) - timesteps_tensor)
        closest_indices = torch.argmin(distances, dim=-1)
        discrete_samples = timesteps_tensor[0, closest_indices]
        return discrete_samples.reshape(B,1,1,1)  # Reshape to [B, 1, 1, 1] for broadcasting
    else:
        raise ValueError(f"Unsupported s_type: {s_type}. Choose from ['logit_normal','uniform'].")

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sample_continue(
        B: int,
        alpha: float = 4.0,
        beta: float = 1.5,
        s_type: str = "logit_normal",
        step: int = 0,
        dynamic_step: int = 1000,
                        ) -> torch.Tensor:
    """
    B: batch size
    m: mean of the logit normal distribution
    s: standard deviation of the logit normal distribution
    s_type: type of sampling, choose from ["logit_normal","uniform"]
    """
    dynamic_step = float(dynamic_step)
    step = float(step)
    if s_type == "uniform":
        t = torch.rand(B) * (1.0 - 0.001) + 0.001
        return t.reshape(B,1,1,1) # Reshape to [B, 1, 1, 1] for broadcasting
    elif s_type == "logit_normal":
        if dynamic_step > 0:
            progress = min(step / dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141593)))
            alpha = 1.0 + (alpha - 1.0) * cosine_decay
            beta = 1.0 + (beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(alpha, beta).sample((B,))
        return t.reshape(B,1,1,1) # Reshape to [B, 1, 1, 1] for broadcasting
    else:
        raise ValueError(f"Unsupported s_type: {s_type}. Choose from ['logit_normal','uniform'].")


def get_ts_next(ts: torch.Tensor, t_steps: torch.Tensor) -> torch.Tensor:
    """
    ts:       arbitrary shape, any discrete dtype
    t_steps:  -D tensor of unique, ascending time steps
    returns:  tensor of same shape / dtype as ts containing the next time step
               (with wrap-around)
    """
    t_steps = torch.as_tensor(t_steps, device=ts.device).flatten().sort()[0]

    # ---- map value -> position ----
    # convert ts to the same dtype as t_steps (usually int64)
    ts_int = ts.to(t_steps.dtype)

    # searchsorted does the mapping in one vectorised call
    indices = torch.searchsorted(t_steps, ts_int.flatten())           # 0 .. len-1
    next_idx = (indices + 1) % t_steps.numel()

    ts_next = t_steps[next_idx].reshape(ts.shape).to(ts.dtype)
    return ts_next

def get_sample(x0_all, sample_t, t_steps):
    """
    x0_all: a list of tensors, each tensor has shape (b, h, w, d)
    sample_t: a tensor of shape (b)
    t_steps: a list of discrete timesteps, with length equal to the length of x0_all
    """
    stacked_x0 = torch.stack(x0_all, dim=0)
    s_to_index = {value.item(): idx for idx, value in enumerate(t_steps)}
    b = sample_t.shape[0]
    indices = torch.tensor([s_to_index[value.item()] for value in sample_t])
    sample = stacked_x0[indices, torch.arange(b)]
    return sample  # shape (b, h, w, d)

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def ckpt_path_to_lora_path(ckpt_path: str, l_type="gen") -> str:
    base_dir, filename = os.path.split(ckpt_path)
    lora_dir = base_dir.replace('torch_units', f'lora_{l_type}')
    lora_path = os.path.join(lora_dir, filename)
    lora_path = lora_path.replace('.pt','')
    return lora_path

def ckpt_path_to_accelerator_path(ckpt_path: str, num = 1) -> str:
    base_dir, filename = os.path.split(ckpt_path)
    opt_dir = base_dir.replace('torch_units', 'accelerator_' + str(num))
    opt_path = os.path.join(opt_dir, filename)
    opt_path = opt_path.replace('.pt','')
    return opt_path

