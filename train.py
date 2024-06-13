import os
import re
import json
import pickle
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import TextDataset
from model import TransformerModel
from utils import get_vocab_list


def train(rank, cfg, num_gpus):
    # Set device based on rank and available GPUs
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Define your model
    embedding_size = cfg.embedding_size
    nhead = cfg.nhead
    nhid = cfg.nhid
    nlayers = cfg.nlayers
    train_file = cfg.train_file
    val_file = cfg.val_file

    with open(train_file, 'rb') as f:
            train_data = pickle.load(f)

    with open(val_file, 'rb') as f:
            test_data = pickle.load(f)

    vocab_list = ["<MASK>", "<PAD>", "<SOS>", "<EOS>"]

    vocab_list = get_vocab_list(train_dataset, vocab_list)
    vocab_list = get_vocab_list(val_file, vocab_list)

    vocab_size = len(vocab_size)

    train_dataset = TextDataset(train_data,
                                vocab_list)
    test_dataset = TextDataset(test_data, 
                               vocab_list)

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
