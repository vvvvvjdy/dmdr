import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "DyJiang/dmdr"
ckpt_name = "sdxl-dmdr-1step-odeinit-refl-dfnclip-hps21-unet/unet_sdxl_base_1.0_1step_res1024_fp32.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model.
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet")
unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cpu"))
pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device=device, dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
prompt = "A photo of cat"
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399]).images[0]
plt.imshow(image)
plt.axis('off')
plt.show()
image.save("sdxl_1step_output.png")