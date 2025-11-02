# Machine Learning Opeartions - Project 2
A containerized machine learning project for training a DistilBERT model on the MRPC (Microsoft Research Paraphrase Corpus) task from the GLUE benchmark. This project demonstrates MLOps best practices including Docker containerization, experiment tracking with Weights & Biases, and reproducible training pipelines.

## 🚀 Quick Start

### Prerequisites

- Docker installed on your machine ([Get Docker](https://docs.docker.com/get-docker/))
- Weights & Biases account and API key ([Sign up here](https://wandb.ai/signup))

### Build and Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/karimdrw/mlops-project2.git
   cd mlops-project2
   ```

2. **Build the Docker image**
   ```bash
   docker build -t mrpc-training .
   ```
   
   *Note: Building takes approximately 5-25 minutes depending on your machine and network speed. The final image is around 12.5GB.*

3. **Run training with default hyperparameters**
   ```bash
   docker run -e WANDB_API_KEY=your_api_key_here mrpc-training
   ```
   
   Replace `your_api_key_here` with your actual Weights & Biases API key.

## 🔧 Custom Training Configuration

You can override the default hyperparameters by passing arguments to the training script:

```bash
docker run -e WANDB_API_KEY=your_api_key_here mrpc-training \
  python train.py \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --batch_size 32 \
  --epochs 5 \
  --checkpoint_dir /app/models
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 5e-5 | Learning rate |
| `--weight_decay` | 0.0 | Weight decay for regularization |
| `--warmup_steps` | 0 | Number of warmup steps for learning rate scheduler |
| `--batch_size` | 24 | Training batch size |
| `--epochs` | 3 | Number of training epochs |
| `--checkpoint_dir` | /app/models | Directory to save model checkpoints |
| `--wandb_project` | mrpc-docker-training | W&B project name |
| `--wandb_run_name` | auto-generated | Custom name for W&B run |

## 📊 Monitoring Training

Training runs are automatically logged to Weights & Biases. After starting a run, you'll see a link in the console output to view your experiment dashboard with:
- Training and validation metrics (accuracy, F1 score)
- Hyperparameter configurations
- System metrics (GPU/CPU usage, memory)

## 🐳 Running on GitHub Codespaces

This project works on GitHub Codespaces with the following recommendations:

1. **Machine size**: Use at least a 4-core, 16GB RAM instance to avoid OOM errors
2. **Timeout**: Set container timeout to 60+ minutes for full training completion

```bash
# In your Codespace terminal
docker build -t mrpc-training .
docker run -e WANDB_API_KEY=your_api_key_here mrpc-training
```

## 📁 Project Structure

```
.
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── train.py               # Main training script with CLI
├── GLUEDataModule.py      # PyTorch Lightning data module
├── GLUETransformer.py     # Model wrapper for training
└── README.md              # This file
```

## 🔍 Technical Details

- **Model**: DistilBERT (distilbert-base-uncased)
- **Task**: MRPC paraphrase detection (binary classification)
- **Framework**: PyTorch Lightning
- **Base Image**: Python 3.11-slim
- **PyTorch**: CPU-only version (torch 2.0.0)
- **Experiment Tracking**: Weights & Biases

## ⚠️ Known Issues

- Two deprecation warnings from W&B may appear during training (see [wandb/wandb#10662](https://github.com/wandb/wandb/issues/10662))
- Training on CPU is significantly slower than GPU (expected behavior)

## 📝 License

This project is part of the MLOps course at [Your University]. Feel free to use for educational purposes.

## 🤝 Contributing

This is a course project, but suggestions and feedback are welcome! Please open an issue for any problems or improvements.

---

**Author**: Karim Darwiche  
**Course**: Machine Learning Operations  
**Date**: October 2025
