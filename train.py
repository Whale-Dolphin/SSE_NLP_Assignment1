import os
import random
import pickle
from typing import Optional

from functools import wraps
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
import hydra

from dataset import TextDataset 
from model import MaskedLanguageModel 
from utils import get_vocab_list, clean_text

def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

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

def load_vocab_list(vocab_list_path):
    if os.path.exists(vocab_list_path):
        with open(vocab_list_path, 'rb') as f:
            vocab_list = pickle.load(f)
    else:
        vocab_list = []
    return vocab_list

def save_vocab_list(vocab_list, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(vocab_list, f)


def train(rank, cfg, world_size):
    if world_size > 1:
        setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Load data
    with open(cfg.train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(cfg.val_file, 'rb') as f:
        val_data = pickle.load(f)

    # Clean data
    train_data = [clean_text(text) for text in train_data]
    val_data = [clean_text(text) for text in val_data]

    # Load or create vocabulary list
    vocab_list = load_vocab_list(cfg.vocab_list_path)
    special_tokens = ["<MASK>", "<PAD>", "<SOS>", "<EOS>"]
    vocab_list = get_vocab_list(special_tokens, vocab_list)
    vocab_list = get_vocab_list(train_data, vocab_list)
    vocab_list = get_vocab_list(val_data, vocab_list)
    vocab_size = len(vocab_list)
    save_vocab_list(vocab_list, cfg.vocab_list_path)

    # print(type(train_data))

    # Create datasets and data loaders
    train_dataset = TextDataset(train_data, vocab_list)
    val_dataset = TextDataset(val_data, vocab_list)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, shuffle=train_sampler is None)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=val_sampler is None)

    # 初始化模型
    model = MaskedLanguageModel(vocab_size, cfg.embedding_size, cfg.nhead, cfg.nhid, cfg.nlayers).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 定义损失函数和优化器
    pad_idx = vocab_list.index("<PAD>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # 如果有可用检查点，则加载
    start_epoch = 0
    if cfg.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, cfg.checkpoint_path)

    scaler = torch.cuda.amp.GradScaler()  # 混合精度

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for i, (mask_indices, attention_mask, text_indices) in enumerate(train_dataloader):
            mask_indices, attention_mask, text_indices = mask_indices.to(device), attention_mask.to(device), text_indices.to(device)

            # print(mask_indices.shape)
            # print(mask_indices[0])

            with torch.cuda.amp.autocast(): 
                outputs = model(mask_indices, attention_mask)
                loss = criterion(outputs.view(-1, vocab_size), text_indices.view(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 100 == 0 and rank == 0:
                print(f'第 {epoch+1}/{cfg.epochs} 轮, 步骤 {i+1}/{len(train_dataloader)}, 损失: {loss.item()}')

        # 验证步骤
        if rank == 0:
            val_loss = validate(model, val_dataloader, criterion, device)
            print(f'第 {epoch+1}/{cfg.epochs} 轮, 验证损失: {val_loss}')

        # 保存检查点
        if rank == 0 and cfg.checkpoint_path:
            save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)

    if world_size > 1:
        cleanup()

@no_grad
def validate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for mask_indices, attention_mask, text_indices in val_dataloader:
        mask_indices, attention_mask, text_indices = mask_indices.to(device), attention_mask.to(device), text_indices.to(device)

        # 前向传播
        outputs = model(mask_indices, attention_mask)
        loss = criterion(outputs.view(-1, model.linear.out_features), text_indices.view(-1))
        total_loss += loss.item()

    average_loss = total_loss / len(val_dataloader)
    return average_loss

@hydra.main(version_base="1.3", config_path="./configs", config_name="transformer.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print('初始化训练进程..')

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() and cfg.use_cuda else 0

    if n_gpus > 1:
        print(f"使用 {n_gpus} 个 GPU 进行训练。")
        mp.spawn(train, nprocs=n_gpus, args=(cfg, n_gpus))
    elif n_gpus == 1:
        print("使用单个 GPU 进行训练。")
        train(0, cfg, n_gpus)
    else:
        print("CUDA 不可用，在 CPU 上训练。")
        train(0, cfg, 1)

if __name__ == "__main__":
    main()
