
CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60210
NUM_PROCS=8
PYTHON_SCRIPT="train_sd3_srpo_deepspeed.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "sd35l_4step/" \
    --exp-name     "srpo_res1024_ode_lora_s5_dfnclip_hps21" \
    --sample-steps      20 \
    --checkpoint-steps  50 \
    --max-train-steps   2301 \
    --cold-start-iter   2000 \
    --pretrained_model   "stabilityai/stable-diffusion-3.5-large" \
    --sample-num-steps  4 \
    --shift  5 \
    --sample-type "ode" \
    --use-lora-gen 2 \
    --lora-rank 32 \
    --lora-scale-f  2.0 \
    --lora-scale-r  0.5\
    --data-path-train "t2i-2m-filtered-prompts.txt" \
    --resolution 1024 \
    --seed   30 \
    --mixed-precision "bf16" \
    --batch-size 2 \
    --batch-size-test 2 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 1e-7 \
    --learning-rate-gui 5e-9 \
    --s-type-gen "uniform" \
    --s-type-gui "logit_normal" \
    --dynamic-step 2000 \
    --gen-a 1.0 \
    --gen-b 1.0 \
    --gui-a 4.0 \
    --gui-b 1.5 \
    --cfg-r 4.0 \
    --cfg-f 1.0 \
    --ratio-update 5.0 \
    --use-8bit-adam \
    --reward-clip-weight 0.1 \
    --reward-hps-weight 5 \
    --magic-prompt "Realistic photo" \
    --clip-reward-model "Apple_CLIP-H-14-224.bin" \
    --hpsv2-reward-model 'HPS_v2.1_compressed.pt' \
    --srpo-k 0.1 \
    --resume-ckpt "/torch_units/step-2000.pt"


