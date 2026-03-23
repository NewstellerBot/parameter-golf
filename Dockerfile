FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /workspace/parameter-golf

# SSH server for RunPod
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd /root/.ssh && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    rm -rf /var/lib/apt/lists/*

# Python deps not in NGC image
RUN pip install --no-cache-dir sentencepiece huggingface-hub datasets tiktoken

# Copy our code
COPY train_gpt.py .
COPY data/ data/
COPY run_experiments.sh .
RUN chmod +x run_experiments.sh

# Download full dataset during build (cached in image)
RUN python3 data/cached_challenge_fineweb.py --variant sp1024

# Start script: setup SSH keys from RunPod env, start sshd, stay alive
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
