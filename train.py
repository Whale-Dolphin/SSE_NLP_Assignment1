import os
import re
import json
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import TransformerModel


def train(rank, cfg, num_gpus):
    # Set device based on rank and available GPUs
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Define your model
    vocab_size = cfg.vocab_size
    embedding_size = cfg.embedding_size
    nhead = cfg.nhead
    nhid = cfg.nhid
    nlayers = cfg.nlayers
    model = TransformerModel(vocab_size, embedding_size, nhead, nhid, nlayers).to(device)

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Load your data (this is just a placeholder, replace with your actual data loading)
    train_dataloader = DataLoader(...)
    for epoch in range(cfg.epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{cfg.epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}')

    

@hydra.main(version_base="1.3", config_path="./configs", config_name="transformer.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print('Initializing Training Process..')

    n_gpus = 0
    if torch.cuda.is_available():
        print("CUDA is available.")
        
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpus}")

        if n_gpus > 1:
            print("Using multiple GPUs for training.")
    else:
        print("CUDA is not available, training on CPU.")

    mp.spawn(train, nprocs=n_gpus, args=(cfg, n_gpus))

if __name__ == "__main__":
    main()
