#!/usr/bin/env python3
"""
Training script for MRPC paraphrase detection using DistilBERT
Converts Project 1 notebook code to a CLI-based script
"""
import argparse
from datetime import datetime
import os
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
import wandb
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT on MRPC paraphrase detection")
    
    # Hyperparameters
    parser.add_argument("--lr", "--learning_rate", type=float, default=2e-5, 
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (default: 0.0)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps (default: 0)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    
    # Paths and configuration
    parser.add_argument("--checkpoint_dir", type=str, default="./models",
                        help="Directory to save model checkpoints (default: ./models)")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Pretrained model name (default: distilbert-base-uncased)")
    parser.add_argument("--task_name", type=str, default="mrpc",
                        help="GLUE task name (default: mrpc)")
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="mrpc-docker-training",
                        help="W&B project name (default: mrpc-docker-training)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    
    # Setup data module
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.task_name,
        train_batch_size=args.batch_size,
    )
    dm.setup("fit")
    
    # Setup model
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.batch_size,
    )

    # Setup logger
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("W&B login successful via environment variable")
    else:
        print("W&B API key not found in environment variables, please set it to enable logging.")
        exit(1)

    run_name = args.wandb_run_name or f"docker-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        log_model=False
    )

    print(f"W&B logging enabled: {args.wandb_project}/{run_name}")
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        default_root_dir=args.checkpoint_dir,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training with hyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*60}\n")
    
    trainer.fit(model, datamodule=dm)
    
    wandb.finish()
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()