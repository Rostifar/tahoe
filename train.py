import os
import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import tokenizer as tok
from data import (
    save_checkpoint,
    load_checkpoint,
    load_batches
)
from loss import (
    cross_entropy
)
from optim import (
    cosine_lr_scheduler,
    clip_grads,
    AdamW
)
from config import Config, DataConfig, ExperimentConfig
from transformer import Transformer

def get_checkpoint_path(iteration: int, exp_config: ExperimentConfig):
    file = f"{exp_config.name}_{iteration}.pt"
    return os.path.join(exp_config.ckpt_path, file)


def get_model(vocab_size: int, exp_config: ExperimentConfig) -> nn.Module:
    return Transformer(
        vocab_size=vocab_size, 
        context_length=exp_config.context_length, 
        num_layers=exp_config.num_layers,
        d_model=exp_config.d_model,
        num_heads=exp_config.num_heads,
        theta=exp_config.theta,
        d_ff=exp_config.d_ff,
        device=exp_config.primary_device,
        dtype=exp_config.get_dtype()
    )


def get_tokenizer(data_config: DataConfig) -> tok.Tokenizer:
    return tok.Tokenizer(
        *tok.load(data_config.vocab), 
        special_tokens=["<|endoftext|>"]
    )


def eval(
    exp_config: ExperimentConfig, 
    val_set: np.array, 
    model: nn.Module, 
    device: torch.device, 
    iteration: int
) -> float:
    model.eval()
    batch_size, context_length = exp_config.batch_size, exp_config.context_length
    running_loss = 0.
    total_tokens = 0
    # enforce a `batch_size` sensible default
    batch_size = max(batch_size, 32)
    with torch.no_grad():
        for batch in load_batches(val_set, batch_size, context_length, device):
            inputs, targets = batch
            logits = model(inputs)
            num_tokens = targets.numel()
            running_loss += cross_entropy(logits, targets).item() * num_tokens
            total_tokens += num_tokens
        print(f"> Validation loss at iteration {iteration}: {running_loss / total_tokens}")
    model.train()
    return running_loss / total_tokens


def train(data_config: DataConfig, exp_config: ExperimentConfig) -> None:
    print(f"> Starting run for {exp_config.name}.")
    run = wandb.init(
        entity="torusai",
        project=exp_config.name,
        config={
            'data': data_config.model_dump(),
            'exp': exp_config.model_dump()
        },
    )
    
    tokenizer = get_tokenizer(data_config)
    print(f"> Vocab Size: {len(tokenizer.vocab)}")
    
    model = get_model(tokenizer.vocab_size, exp_config)
    wandb.watch(model, log="all", log_freq=100)

    optimizer = AdamW(
        params=model.parameters(),
        # Note: may be overwritten by scheduler.
        lr=exp_config.optimizer.lr,
        betas=exp_config.optimizer.betas,
        weight_decay=exp_config.optimizer.weight_decay
    )

    if exp_config.from_ckpt:
        iteration = load_checkpoint(
            data_config.from_ckpt, 
            model, 
            optimizer, 
            exp_config.primary_device
        )
    else:
        iteration = 1

    print(f"> Loading Datasets: {data_config.train_set}, {data_config.val_set}")
    train_set = np.load(data_config.train_set, mmap_mode='r').astype(np.uint16)
    val_set = np.load(data_config.val_set, mmap_mode='r').astype(np.uint16)
    print(f"> Training Set Size (tokens): {len(train_set)}")
    print(f"> Validation Set Size (token): {len(val_set)}\n")

    batch_size, context_length = exp_config.batch_size, exp_config.context_length
    total_batches = len(train_set) // ((batch_size - 1) * (context_length) + context_length + 1)
    
    print(f"> Total Batches: {total_batches}")
    print(f"> Batch Size: {batch_size}")
    print(f"> Context Length: {context_length}")    
    print(f"> Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"> dtype: {exp_config.get_dtype()}")
    print(f"> device: {exp_config.primary_device}")
    print(f"> Loaded checkpoint: {exp_config.from_ckpt}")
    print(f"> Starting Iteration: {iteration}\n")
    if exp_config.max_iter:
        print(f"> Estimated training tokens: {batch_size * context_length * exp_config.max_iter}")

    model.train()
    running_duration = 0.
    batch_iter = load_batches(
        train_set, 
        batch_size, 
        context_length, 
        exp_config.primary_device, 
        iteration
    )
    for batch in tqdm(batch_iter):
        if exp_config.max_iter and iteration > exp_config.max_iter:
            print("Terminating after max iterations...")
            return

        # run this first to replace AdamW LR placeholder
        scheduler = exp_config.scheduler
        if scheduler:
            lr = cosine_lr_scheduler(
                iteration, 
                scheduler.lr_max, 
                scheduler.lr_min, 
                scheduler.t_warmup, 
                scheduler.t_cos
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = exp_config.optimizer.lr
        
        start = time.perf_counter()
        inputs, targets = batch
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        
        clip_grads(model.parameters(), max_grad=exp_config.max_grad)
        optimizer.step()
        optimizer.zero_grad()
       
        end = time.perf_counter()
        running_duration += end - start

        iteration += 1
        avg_duration = running_duration / iteration
        run.log({"loss": loss.item(), "lr": lr, "duration": avg_duration})
        if iteration % 100 == 0:
            print(f"--Update--")
            print(f"Loss[iter={iteration}]={loss.item()}")
            print(f"LR[iter={iteration}]={lr}")
            print(f"Batch={iteration}/{total_batches}")
            print(f"Average Duration={running_duration / (iteration + 1)}\n")

        if iteration % exp_config.ckpt_iter == 0:
            path = get_checkpoint_path(iteration, exp_config)
            save_checkpoint(model, optimizer, iteration, path)

        if exp_config.val_iter and iteration % exp_config.val_iter == 0:
            val_loss = eval(exp_config, val_set, model, exp_config.primary_device, iteration)
            run.log({"val_loss": val_loss})


if __name__ == "__main__":
    configs = Config.from_args()
    train(*configs)
