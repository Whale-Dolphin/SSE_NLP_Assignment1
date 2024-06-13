import os
import re
import json
import pickle
from typing import Optional

import hydra
from functools import wraps
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from dataset import TextDataset
from model import MaskedLanguageModel
from utils import get_vocab_list, split_text

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint at epoch {checkpoint['epoch']}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
        return 0

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def train(rank, cfg, world_size):
    if world_size > 1:
        setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Load data
    with open(cfg.train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(cfg.val_file, 'rb') as f:
        val_data = pickle.load(f)

    # Build vocabulary list
    vocab_list = ["<MASK>", "<PAD>", "<SOS>", "<EOS>"]
    vocab_list = get_vocab_list(train_data, vocab_list)
    vocab_list = get_vocab_list(val_data, vocab_list)
    vocab_size = len(vocab_list)

    # Create datasets
    train_dataset = TextDataset(train_data, vocab_list)
    val_dataset = TextDataset(val_data, vocab_list)

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, shuffle=train_sampler is None)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=val_sampler is None)

    # Initialize the model
    model = MaskedLanguageModel(vocab_size, cfg.embedding_size, cfg.nhead, cfg.nhid, cfg.nlayers).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Load checkpoint if available
    start_epoch = 0
    if cfg.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for i, (inputs, attention_mask, labels) in enumerate(train_dataloader):
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            # Generate source mask for Transformer
            src_mask = model.transformer_encoder.generate_square_subsequent_mask(inputs.size(1)).to(device)

            # Forward pass
            outputs = model(inputs, src_mask)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 and rank == 0:
                print(f'Epoch [{epoch+1}/{cfg.epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}')

        # Validation step
        if rank == 0:
            val_loss = validate(model, val_dataloader, criterion, device)
            print(f'Epoch [{epoch+1}/{cfg.epochs}], Validation Loss: {val_loss}')

        # Save checkpoint
        if rank == 0 and cfg.checkpoint_path:
            checkpoint_path = os.path.join(cfg.checkpoint_path, f"epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    if world_size > 1:
        cleanup()

def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

@no_grad
def validate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for inputs, attention_mask, labels in val_dataloader:
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

        # Generate source mask for Transformer
        src_mask = model.transformer_encoder.generate_square_subsequent_mask(inputs.size(1)).to(device)

        # Forward pass
        outputs = model(inputs, src_mask)
        loss = criterion(outputs.view(-1, model.linear.out_features), labels.view(-1))
        total_loss += loss.item()
    
    average_loss = total_loss / len(val_dataloader)
    return average_loss

@hydra.main(version_base="1.3", config_path="./configs", config_name="transformer.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print('Initializing Training Process..')

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs for training.")
        mp.spawn(train, nprocs=n_gpus, args=(cfg, n_gpus))
    elif n_gpus == 1:
        print("Using single GPU for training.")
        train(0, cfg, n_gpus)
    else:
        print("CUDA is not available, training on CPU.")
        train(0, cfg, 1)

if __name__ == "__main__":
    main()
