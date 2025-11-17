#!/usr/bin/env bash

# --- 1. Distributed-training configuration ---
# NOTE: Replace MASTER_ADDR with the actual IP of the machine that runs this script
MASTER_ADDR="your.master.ip.address"

# List of worker nodes
WORKER_NODES=("worker.ip.address.1" "worker.ip.address.2")
# Total nodes: 1 master + number of workers
NUM_NODES=$((1 + ${#WORKER_NODES[@]}))

# Training script name (needs to be the same on all nodes)
TRAIN_SCRIPT="scripts/multi_node/cold_start_sdxl.sh"

# Directory that contains the training script
REMOTE_DIR="train_t2i/sdxl"

# Conda/venv activation script (change to your env path, the env name is dmdr here)
SOURCE_DIR="/conda/bin/activate"

# --- 2. Launch master node (RANK=0) ---

echo "Starting master node (RANK=0) on ${MASTER_ADDR}..."

# Export env vars required for distributed training
export NODE_RANK=0
export NUM_NODES=${NUM_NODES}
export MASTER_ADDR=${MASTER_ADDR}

# Start training locally
cd ${REMOTE_DIR}
bash ${TRAIN_SCRIPT} &

# --- 3. Launch worker nodes (RANK=1, 2, ...) ---

echo "Starting worker nodes..."

for i in "${!WORKER_NODES[@]}"; do
    NODE_RANK=$((i + 1))   # Worker ranks start from 1
    WORKER_IP=${WORKER_NODES[$i]}

    echo "Launching worker node (RANK=${NODE_RANK}) on ${WORKER_IP}..."

    # Build remote command
    REMOTE_COMMAND="
        export NODE_RANK=${NODE_RANK} && \
        export NUM_NODES=${NUM_NODES} && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        source ${SOURCE_DIR} dmdr && \
        conda activate dmdr && \
        cd ${REMOTE_DIR} && \
        bash ${TRAIN_SCRIPT}
    "

    # Execute via SSH
    ssh  -p 22 ${WORKER_IP} "exec bash -l -c '${REMOTE_COMMAND}'" &
done

# --- 4. Wait for all processes to finish ---
wait

echo "All training processes finished."