

##  🎁 Prepare Dataset and Reward Model

By default, we use text prompts from t2i-2M as our training dataset. We use DFN5B-CLIP-ViT-H-14 and HPSv2.1 as our reward models.

- Download t2i-2M text-prompts dataset from [here](https://huggingface.co/DyJiang/dmdr/blob/main/t2i-2m-filtered-prompts.txt) and rename it to `t2i-2m-filtered-prompts.txt`.
- Download DFN5B-CLIP-ViT-H-14 from [here](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/blob/main/open_clip_pytorch_model.bin) and rename it to `Apple_CLIP-H-14-224.bin`.
- Download HPSv2.1 from [here](https://huggingface.co/xswu/HPSv2/blob/main/HPS_v2.1_compressed.pt) and rename it to `HPS_v2.1_compressed.pt`.




## 🚂 Start Training

### SD3.5-Large 4-step DMDR Training 

#### Single Node (Not recommended, only for quick test and debugging)
```bash
# Cold Start

cd train_t2i/sd3
bash scripts/single_node/cold_start_sd35.sh
```
```bash
# RL + DMD

# DMDR (ReFL)
bash scripts/single_node/dmdr_sd35_refl.sh

# DMDR (SRPO)
bash scripts/single_node/dmdr_sd35_srpo.sh
```

The output directory structure will be like:
```output_dir/
├── checkpoints/
│   ├── accelerator_1/
│   │   └── step-xxx/
│   └── accelerator_2/
│   │   └── step-xxx/
│   └── lora_gen/
│   │   └── step-xxx/
│   └── lora_gui/
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
- `--sample-num-steps`: Number of steps of the few-step diffusion model you want to train
- `--use-lora-gen`: Whether to use LoRA for the few-step diffusion model (enabled by default, set to 0 to disable)
- `--lora-rank`: LoRA rank for the score estimator (the LoRA rank of the few-step diffusion model will be set to twice of this value)
- `--data-path-train`: Change to the path of `t2i-2m-filtered-prompts.txt` you downloaded
- `--batch-size`:  Local batch size ( Total = `batch-size` * number of processes * gradient accumulation steps)
- `--sample-type`: Choose from {`ode`, `sde`} (default: "ode")
- `--resolution`: Image resolution to train (default: 1024 for SD3.5-Large and 512 for SD3-Medium)
- `--clip-reward-model`: Path to the Apple_CLIP-H-14-224.bin model you downloaded
- `--hpsv2-reward-model`: Path to the HPS_v2.1_compressed.pt model you downloaded
- `--resume-ckpt`: Path to the `torch_units/step-xxx.pt` to resume training (only used in the second stage of training)

#### Multi Node (Recommended, by default we use 24 GPUs, and our machine can use SSH to connect to each other)
```bash
cd train_t2i/sd3
bash scripts/multi_node/multi.sh
```
Some options you need to complete or adjust in the bash scripts:

- `MASTER_ADDR`: Your.master.ip.address
- `WORKER_NODES`: List of your worker node ip addresses
- `TRAIN_SCRIPT': Which training script to run, e.g., `scripts/multi_node/dmdr_sd35_refl.sh`
- `REMOTE_DIR`: Directory that contains the python training script 
- `SOURCE_DIR`: Conda/venv activation script (default: `/conda/bin/activate`, the env name is set to dmdr by default)


### SD3-Medium 4-step DMDR Training 


#### Single Node (Not recommended, only for quick test and debugging)
```bash
# Cold Start

cd train_t2i/sd3
bash scripts/single_node/cold_start_sd3.sh
```
```bash
# RL + DMD

# DMDR (ReFL)
bash scripts/single_node/dmdr_sd3_refl.sh

# DMDR (SRPO)
bash scripts/single_node/dmdr_sd3_srpo.sh
```

#### Multi Node (Recommended, by default we use 24 GPUs, and our machine can use SSH to connect to each other)
```bash
cd train_t2i/sd3
bash scripts/multi_node/multi.sh
```