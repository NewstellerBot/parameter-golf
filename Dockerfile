FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace/parameter-golf

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Python deps (from requirements.txt + triton)
RUN pip install --no-cache-dir \
    numpy tqdm sentencepiece huggingface-hub datasets \
    tiktoken setuptools typing-extensions==4.15.0 triton

# Copy our code
COPY train_gpt.py .
COPY data/ data/
COPY run_experiments.sh .
RUN chmod +x run_experiments.sh

# Download full dataset during build (cached in image)
RUN python3 data/cached_challenge_fineweb.py --variant sp1024

# Default: drop into bash so you can run experiments
CMD ["/bin/bash"]
