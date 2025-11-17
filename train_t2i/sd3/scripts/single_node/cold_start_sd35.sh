
CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60210
NUM_PROCS=8
PYTHON_SCRIPT="train_sd3_refl_deepspeed.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "sd35l_4step/" \
    --exp-name     "pretrain_dynacs_res1024_ode_lora_s5" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --max-train-steps   2001 \
    --cold-start-iter   2001 \
    --pretrained_model  "stabilityai/stable-diffusion-3.5-large" \
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
    --batch-size 4 \
    --batch-size-test 2 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 1e-6 \
    --learning-rate-gui 2e-8 \
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



