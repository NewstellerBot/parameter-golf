#!/bin/bash
# RunPod container start script: setup SSH and stay alive

# Setup SSH keys from RunPod environment
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A 2>/dev/null  # generate host keys if missing
service ssh start

echo "=== Parameter Golf container ready ==="
echo "cd /workspace/parameter-golf && RUN_ID=test NUM_LAYERS=19 torchrun --standalone --nproc_per_node=1 train_gpt.py"

# Keep container alive
sleep infinity
