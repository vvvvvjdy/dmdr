CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60210
NUM_PROCS=8
PYTHON_SCRIPT="train_sitxl_refl_deepspeed.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "sit_1step/" \
    --exp-name     "pretrain_res256_nocfg_s1" \
    --sample-steps      1000 \
    --checkpoint-steps  5000 \
    --max-train-steps   20001 \
    --cold-start-iter   20001 \
    --model             "SiT-XL/2" \
    --pretrain-path      "sit_xl_repa_in1k_800ep_nopjhead.pt" \
    --fused-attn \
    --num-steps  1 \
    --shift  1 \
    --lora-rank 32 \
    --lora-scale-f  2.0 \
    --lora-scale-r  0.75\
    --resolution 256 \
    --seed   30 \
    --mixed-precision "fp16" \
    --batch-size 64 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 2e-5 \
    --learning-rate-gui 2e-5 \
    --s-type-gen "uniform" \
    --s-type-gui "logit_normal" \
    --dynamic-step 10000 \
    --gen-a 1.0 \
    --gen-b 1.0 \
    --gui-a 4.0 \
    --gui-b 1.5 \
    --cfg-r 0 \
    --ratio-update 5.0 \





