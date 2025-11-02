# Use Python 3.11 slim image for smaller size and security patches
FROM python:3.11-slim

WORKDIR /app

# install system dependencies (if needed for some packages)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy the requirements
COPY requirements.txt .

# installing python dependencies, including specific CPU-only PyTorch version
RUN pip install --no-cache-dir torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .
COPY GLUEDataModule.py .
COPY GLUETransformer.py .

# Create directory for model checkpoints
RUN mkdir -p /app/models

# env variables
ENV PYTHONUNBUFFERED=1

# default, best results from my first project
CMD ["python", "train.py", \
     "--lr", "5e-5", \
     "--weight_decay", "0", \
     "--warmup_steps", "0", \
     "--batch_size", "24", \
     "--epochs", "3", \
     "--checkpoint_dir", "/app/models" \
    ]
