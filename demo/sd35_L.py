import torch
import matplotlib.pyplot as plt
from peft import PeftModel
from diffusers import StableDiffusion3Pipeline
base_path = "DyJiang/dmdr"
# dmdr with srpo
subfolder = "sd35l-dmdr-4step-srpo-dfnclip-hpsv21-lora"
# dmdr with refl
# subfolder = "sd35l-dmdr-4step-refl-dfnclip-hpsv21-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16
).to(device)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer,
                                             base_path,
                                             subfolder = subfolder
                                              ).to(device=device, dtype=torch.float16)
pipe.scheduler.config['shift'] = 5
prompt = "A photorealistic tiny dragon taking a bath in a teacup, coherent, intricate"
with torch.no_grad():
    sample = pipe(
        prompt=prompt,
        guidance_scale=0,
        height=1024,
        width=1024,
        num_inference_steps=4,
    ).images[0]
plt.imshow(sample)
plt.axis('off')
plt.show()
sample.save("sd35l_dmdr_4step_example.png")
