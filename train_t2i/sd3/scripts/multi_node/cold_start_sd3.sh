export NCCL_CROSS_NIC=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONUNBUFFERED=1
unset NCCL_DEBUG



CONFIG_FILE="configs/multi_node.yaml"
MAIN_PORT=19999
NUM_PROCS=24
PYTHON_SCRIPT="train_sd3_refl_deepspeed.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    --num_machines      ${NUM_NODES} \
    --machine_rank      ${NODE_RANK} \
    --main_process_ip   ${MASTER_ADDR} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "sd3m_4step/" \
    --exp-name     "pretrain_dynacs_res1024_ode_lora_s3" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --max-train-steps   2001 \
    --cold-start-iter   2001 \
    --pretrained_model  "stabilityai/stable-diffusion-3-medium-diffusers" \
    --sample-num-steps  4 \
    --shift  3 \
    --sample-type "ode" \
    --use-lora-gen 2 \
    --lora-rank 32 \
    --lora-scale-f  2.0 \
    --lora-scale-r  0.5\
    --data-path-train "t2i-2m-filtered-prompts.txt" \
    --resolution 512 \
    --seed   30 \
    --mixed-precision "bf16" \
    --batch-size 8 \
    --batch-size-test 2 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 1e-6 \
    --learning-rate-gui 5e-7 \
    --s-type-gen "uniform" \
    --s-type-gui "logit_normal" \
    --dynamic-step 2000 \
    --gen-a 1.0 \
    --gen-b 1.0 \
    --gui-a 4.0 \
    --gui-b 1.5 \
    --cfg-r 7.0 \
    --cfg-f 1.0 \
    --ratio-update 5.0 \
    --use-8bit-adam \


