## 🎃 Discussion
In our ImageNet experiments, we mainly focus on class-conditional generation using SiT as the backbone model, because there is clearly a trend to use DiT based models to conduct the research on ImageNet image generation.

During our experiments on ImageNet, We have discovered several interesting phenomena and we hope to share them.

1. The role of cfg in DMD: On more complex tasks such as t2i, we found that the real-score estimator of DMD requires a large CFG to work (see **[Decoupled DMD](https://arxiv.org/abs/2511.22677)**). However, on ImageNet, the phenomenon is somewhat different. It works without CFG, but as we gradually increase the CFG, a clear trade-off between quality and diversity can be observed

2. The role of RL in DMDR: By incorporating RL (where we use DINOv2 as the Reward model and directly maximize the classification accuracy score), a significant trade-off between quality and diversity can also be observed.


### 1-step SiT DMDR Results on ImageNet 256x256
| CFG | with RL | FID   | Recall | Inception Score | Precision |
|-----|---------|-------|--------|-----------------|-----------|
| No  | No      | 2.13  | 0.64   | 232.15          | 0.77      |
| 1.5 | No      | 5.20  | 0.48   | 387.24          | 0.88      |
| 4   | No      | 15.70 | 0.11   | 453.65          | 0.86      |
| No  | Yes     | 6.95  | 0.40   | 416.01          | 0.90      |
| 1.5 | Yes     | 13.43 | 0.24   | 494.43          | 0.93      |


## 🛠️ Environment Setup
```bash
git clone https://github.com/vvvvvjdy/dmdr.git
conda create -n dmdr python=3.12 -y
conda activate dmdr
pip install -r requirements.txt
```
****
 
## 🎓  Prepare Model

We use SiT-XL trained with [REPA](https://github.com/sihyun-yu/REPA) to initialize our few-step SiT model.
Just run the following command to download the pretrained SiT-XL-REPA model (You need to change the `save_dir` in the python scripts else the ckpt would be put into `sit_weights/`)
```bash
cd train_cc/sit
python convert_weight/sit_convert.py
````



##  🚈 Start Training

### SiT-XL 1-step DMDR Training 

#### Single Node (Recommended)
```bash
# Cold Start

cd train_cc/sit
bash scripts/cold_start_sitxl.sh
```
```bash
# RL + DMD

# DMDR (ReFL)
bash scripts/dmdr_sitxl_refl.sh
```

The output directory structure will be like:
```output_dir/
├── checkpoints/
│   ├── accelerator_1/
│   │   └── step-xxx/
│   └── accelerator_2/
│   │   └── step-xxx/
│   └── torch_units/
│   │   └── step-xxx.pt/
├── loss_logs/
│   │   └── loss_gen_log.jsonl
│   │   └── loss_gui_log.jsonl
├── samples/
│   │   └── step xxx/
├── args.json
└── log.txt
```

Some options you need to complete or adjust in the bash scripts:

- `--output-dir`: Base directory to save checkpoints, samples, and logs
- `--num-steps`: Number of steps of the few-step diffusion model you want to train
- `--use-lora-gen`: Whether to use LoRA for the few-step diffusion model (disable by default)
- `--lora-rank`: LoRA rank for the score estimator (the LoRA rank of the few-step diffusion model will be set to twice of this value)
- `--batch-size`:  Local batch size ( Total = `batch-size` * number of processes * gradient accumulation steps)
- `--encoder-type`: We now only support `dinov2` (set to None if do not use RL)
- `--resume-ckpt`: Path to the `torch_units/step-xxx.pt` to resume training (only used in the second stage of training)

#

