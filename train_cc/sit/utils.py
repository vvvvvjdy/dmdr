import torch
import hashlib
import os
from torchvision.transforms import transforms
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import logging
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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

class DINOv2ProcessorWithGrad(BaseImageProcessor):
    def __init__(
        self,
        res,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.crop_size = res
        self.transforms = transforms.Compose([
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.Resize(size=(self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        ])

    def preprocess(
        self,
        images: ImageInput,
        **kwargs,
    ):
        images = self.transforms(images)
        return images

def ckpt_path_to_accelerator_path(ckpt_path: str, num = 1) -> str:
    base_dir, filename = os.path.split(ckpt_path)
    opt_dir = base_dir.replace('torch_units', 'accelerator_' + str(num))
    opt_path = os.path.join(opt_dir, filename)
    opt_path = opt_path.replace('.pt','')
    return opt_path
