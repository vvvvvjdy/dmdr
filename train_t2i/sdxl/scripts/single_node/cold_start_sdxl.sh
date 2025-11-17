CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60210
NUM_PROCS=8
PYTHON_SCRIPT="train_sdxl_refl_deepspeed.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "sdxl_1step/" \
    --exp-name      "pretrain_odeinit_res1024_unet_399" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --max-train-steps   3001 \
    --cold-start-iter   3001 \
    --pretrained_model  "stabilityai/stable-diffusion-xl-base-1.0" \
    --sample-num-steps  1 \
    --shift  1 \
    --use-lora-gen 0 \
    --lora-rank 32 \
    --lora-scale-f  2.0 \
    --lora-scale-r  0.5\
    --data-path-train "t2i-2m-filtered-prompts.txt" \
    --resolution 1024 \
    --seed   30 \
    --mixed-precision "fp16" \
    --batch-size 8 \
     --batch-size-test 2 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 1e-7 \
    --learning-rate-gui 1e-7 \
    --s-type-gen "uniform" \
    --s-type-gui "uniform" \
    --dynamic-step 0 \
    --gen-a 1.0 \
    --gen-b 1.0 \
    --gui-a 4.0 \
    --gui-b 1.5 \
    --cfg-r 7.0 \
    --cfg-f 1.0 \
    --ratio-update 5.0 \
    --use-8bit-adam \


