import torch
from repa import SiT_models
from  repa_utils import download_model
import os
# Download the pretrained weights of REPA and convert to the original SiT implementation

block_kwargs = {"fused_attn": False, "qk_norm": False}
model = SiT_models["SiT-XL/2"](
    input_size=32,
    num_classes=1000,
    use_cfg=True,
    z_dims = [768],
    encoder_depths=8,
    **block_kwargs
).to("cpu")

save_dir = 'sit_weights/'
state_dict = download_model('last.pt', save_dir)
model.load_state_dict(state_dict, strict=False)
model.eval()
model.projectors = None
print(model)

torch.save(model.state_dict(), os.path.join(save_dir,'sit_xl_repa_in1k_800ep_nopjhead.pt'))
print('save ckpt at: ',os.path.join(save_dir,'sit_xl_repa_in1k_800ep_nopjhead.pt'))

