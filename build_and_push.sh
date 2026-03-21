#!/bin/bash
set -e

# Build and push Docker image to Docker Hub
# Usage: bash build_and_push.sh <dockerhub_username>
# Example: bash build_and_push.sh newstellerbot

USERNAME=${1:?Usage: bash build_and_push.sh <dockerhub_username>}
IMAGE="${USERNAME}/parameter-golf:latest"

echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" .

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo ""
echo "Done! Use on RunPod:"
echo "  Image: ${IMAGE}"
echo "  Docker Command: bash"
echo ""
echo "Then inside the pod:"
echo "  cd /workspace/parameter-golf"
echo "  RUN_ID=test NUM_LAYERS=19 torchrun --standalone --nproc_per_node=\$NGPU train_gpt.py"
