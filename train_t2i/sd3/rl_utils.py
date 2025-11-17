
from torchvision.transforms import transforms
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
import torch


class CLIPImageProcessorTensor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        crop_size: int = 256,
        model_type: str = "siglip",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if model_type == "siglip":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif model_type == "clip":
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std = [0.26862954, 0.26130258, 0.27577711]
        self.crop_size = crop_size
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str = "pt",
        **kwargs,
    ):
        images = self.transforms(images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

def clip_score(img_feature, text_feature, logit_scale=1, logit_bias=0):
    """
    Calculate the CLIP score between image and text features.
    :param img_feature: Image feature tensor.
    :param text_feature: Text feature tensor.
    :return: CLIP score.
    """
    img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)
    text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
    logits_per_image = torch.matmul(img_feature, text_feature.t())
    logits_per_image = logits_per_image * logit_scale + logit_bias
    score = torch.diagonal(logits_per_image)
    return score





