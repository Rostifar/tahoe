import os
import time
import yaml
import wandb
import torch
import argparse
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
from typing import Literal
from pydantic import BaseModel
from transformer import Transformer, decode


class Config(BaseModel):
    vocab: str
    train_set: str
    val_set: str

    batch_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    theta: float

    device: Literal["cpu", "mps", "cuda"]
    dtype: Literal["fp16", "fp32"]

    lr_max: float
    lr_min: float
    t_warmup: int
    t_cos: int

    ckpt_iter: int
    ckpt_path: str
    run_id: str | None = None
    val_iter: int | None = None
    max_iter: int | None = None
    from_ckpt: str | None = None

    betas: tuple[float, float] = (0.99, 0.999)
    weight_decay: float = 1e-2
    max_grad: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        return cls(**data)

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> "Config":
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--config", type=str, help="Path to model config file")
        parsed = parser.parse_args(args)

        if parsed.config:
            config_data = yaml.safe_load(open(parsed.config))
            if "betas" in config_data and isinstance(config_data["betas"], list):
                config_data["betas"] = tuple(config_data["betas"])
        else:
            config_data = {}

        cli_overrides = {
            k.replace("-", "_"): v
            for k, v in vars(parsed).items()
            if v is not None and k != "config"
        }

        if "betas" in cli_overrides and isinstance(cli_overrides["betas"], str):
            b1, b2 = cli_overrides["betas"].split(",")
            cli_overrides["betas"] = (float(b1), float(b2))

        config_data.update(cli_overrides)
        return cls(**config_data)

    def to_yaml(self, path: str) -> None:
        data = {k: (list(v) if k == "betas" else v) for k, v in self.__dict__.items()}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Invalid dtype: {dtype}.")

def get_checkpoint_path(iteration: int, config: Config):
    if config.run_id:
        file = f"{config.run_id}_{iteration}.pt"
    else:
        file = "iteration.pt"
    return os.path.join(config.ckpt_path, file)

def eval(config: Config, val_set: np.array, model: nn.Module, device: torch.device, iteration: int):
    model.eval()
    batch_size, context_length = config.batch_size, config.context_length
    running_loss = 0.
    batches = 0
    for batch in load_batches(val_set, batch_size, context_length, device):
        inputs, targets = batch
        logits = model(inputs)
        running_loss += cross_entropy(logits, targets).item()
        batches += 1
    print(f"Validation loss at iteration {iteration}: {running_loss / batches}")
    model.train()

def train(config: Config) -> None:
    print("--Initializing wandb--")
    run = wandb.init(
        entity="torusai",
        project=config.run_id,
        config=config.model_dump(),
    )

    print("--Loading Tokenizer--")
    tokenizer = tok.Tokenizer(
        *tok.load(config.vocab), 
        special_tokens=["<|endoftext|>"]
    )
    print(f"> Vocab Size: {len(tokenizer.vocab)}")
    
    device = torch.device(config.device)
    dtype = get_dtype(config.dtype)
    
    # create transformer
    model = Transformer(
        vocab_size=tokenizer.vocab_size, 
        context_length=config.context_length, 
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        theta=config.theta,
        d_ff=config.d_ff,
        device=device,
        dtype=dtype
    )
    wandb.watch(model, log="all", log_freq=100)

    optimizer = AdamW(
        params=model.parameters(),
        # placeholder value
        lr=config.lr_max,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    if config.from_ckpt:
        iteration = load_checkpoint(config.from_ckpt, model, optimizer)
    else:
        iteration = 1

    print("--Loading Datasets--")
    train_set = np.load(config.train_set, mmap_mode='r').astype(np.uint16)
    val_set = np.load(config.val_set, mmap_mode='r').astype(np.uint16)
    print(f"> Training Set Size (tokens): {len(train_set)}")
    print(f"> Validation Set Size (token): {len(val_set)}\n")

    batch_size, context_length = config.batch_size, config.context_length
    total_batches = len(train_set) // ((batch_size - 1) * (context_length) + context_length + 1)
    
    print(f"--Run Stats--")
    print(f"> Total Batches: {total_batches}")
    print(f"> Batch Size: {batch_size}")
    print(f"> Context Length: {context_length}")    
    print(f"> Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"> dtype: {dtype}")
    print(f"> device: {device}")
    print(f"> Loaded checkpoint: {config.from_ckpt}")
    print(f"> Starting Iteration: {iteration}\n")

    model.train()
    running_duration = 0.
    for batch in tqdm(load_batches(train_set, batch_size, context_length, device, iteration)):
        if config.max_iter and iteration > config.max_iter:
            print("Terminating after max iterations...")
            return

        # run this first to replace AdamW LR placeholder
        lr = cosine_lr_scheduler(iteration, config.lr_max, config.lr_min, config.t_warmup, config.t_cos)
        for group in optimizer.param_groups:
            group["lr"] = lr
        
        start = time.perf_counter()
        inputs, targets = batch
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        
        clip_grads(model.parameters(), max_grad=config.max_grad)
        optimizer.step()
        optimizer.zero_grad()
       
        end = time.perf_counter()
        running_duration += end - start

        iteration += 1
        run.log({
            "loss": loss, 
            "lr": lr, 
            "duration": running_duration / (iteration + 1)},
        )
        if iteration % 100 == 0:
            print(f"--Update--")
            print(f"Loss[iter={iteration}]={loss}")
            print(f"LR[iter={iteration}]={lr}")
            print(f"Batch={iteration+1}/{total_batches}")
            print(f"Average Duration={running_duration / (iteration + 1)}\n")
        
        if iteration % 500 == 0:
            response = decode(
                model=model,
                prompt=tokenizer.encode("Ron said"),
                stop_token=tokenizer.encode("<|endoftext|>")[0],
                max_tokens=256,
                temperature=1.0,
                top_p=0.9,
            )
            print(f"Generating response for `Ron said`: {tokenizer.decode(response)}")

        if iteration % config.ckpt_iter == 0:
            path = get_checkpoint_path(iteration, config)
            save_checkpoint(model, optimizer, iteration, path)

        if config.val_iter and iteration % config.val_iter == 0:
            eval(config, val_set, model, device, iteration)

if __name__ == "__main__":
    config = Config.from_args()
    train(config)
