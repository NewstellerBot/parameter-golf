FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /workspace/parameter-golf

# Only install what the NGC image doesn't already have
RUN pip install --no-cache-dir sentencepiece huggingface-hub datasets tiktoken

# Copy our code
COPY train_gpt.py .
COPY data/ data/
COPY run_experiments.sh .
RUN chmod +x run_experiments.sh

# Download full dataset during build (cached in image)
RUN python3 data/cached_challenge_fineweb.py --variant sp1024

# Default: drop into bash so you can run experiments
CMD ["/bin/bash"]
