import yaml
import argparse
from data import (
    yield_batch,
    save_checkpoint,
    load_checkpoint
)
from optim import (
    cosine_lr_scheduler,
    clip_grads,
    AdamW
)
from typing import Literal
from transformer import Transformer
from dataclasses import dataclass, field

"""
# Training Loop Requirements

## Args
* vocab: str (path to vocabulary)
* train_set: str (path to training set)
* val_set: str (path to validation set)

* batch_size: int (batch size of training iterations)
* context_length: int (max context length)
* num_layers: int (number of attention layers)
* d_model: int (dimension of embedding space)
* num_heads: int (number of heads for multi-head attention)
* d_ff: int (dimension of feed-forward transformation)
* theta: float (RoPE angle)

* device: 'cpu' | 'mps' | 'cuda' (device for training)
* dtype: 'fp16' | 'fp32' (data type for training)

* lr_max: float (max learning rate for the model)
* lr_min: float (min learning rate for the model)
* t_warmup: int (iterations for LR scheduler warmup)
* t_cos: int (iterations for LR scheduler cosine decay)

* betas: beta values for AdamW (default=0.99,0.999)
* weight_decay: weight decay value for AdamW (default=1e-2)
* max_grad: float (maximum grad magnitude for gradient clipping)

* ckpt_iter: int (number of iterations between checkpoints)
* ckpt_path: str (path for model checkpointing)
* from_ckpt: str | None (checkpoint to start from)

## Logic
1. Load model and optimizer from checkpoints. Otherwise, initialize model.
2. mmap `train_set` and `val_set` before entering training loop.
3. Enter training loop:
    1. Sample token batch of size `(batch_size, context_length)` from training set.
    2. Compute forward pass and loss.
    3. Backprop, step, and update LR sechduler.
    4. Every `ckpt_iter` iterations, save checkpoint.
    5. Every `val_iter` iterations, evaluate loss on training set.

## Extensions
- [ ] Connect to wandb
"""

@dataclass
class TrainingConfig:
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

    betas: tuple[float, float] = (0.99, 0.999)
    weight_decay: float = 1e-2
    max_grad: float = 1.0

    ckpt_iter: int = 1000
    ckpt_path: str = "./checkpoints"
    from_ckpt: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        return cls(**data)

    @classmethod
    def from_args(cls, args: list[str] | None = None) -> "TrainingConfig":
        parser = argparse.ArgumentParser(description="Training configuration")

        # Data paths
        parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
        parser.add_argument("--vocab", type=str, default=None, help="Path to vocabulary")
        parser.add_argument("--train-set", type=str, default=None, help="Path to training set")
        parser.add_argument("--val-set", type=str, default=None, help="Path to validation set")

        # Model architecture
        parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
        parser.add_argument("--context-length", type=int, default=None, help="Max context length")
        parser.add_argument("--num-layers", type=int, default=None, help="Number of attention layers")
        parser.add_argument("--d-model", type=int, default=None, help="Embedding dimension")
        parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
        parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward dimension")
        parser.add_argument("--theta", type=float, default=None, help="RoPE angle")

        # Hardware
        parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="Device")
        parser.add_argument("--dtype", choices=["fp16", "fp32"], default=None, help="Data type")

        # Learning rate schedule
        parser.add_argument("--lr-max", type=float, default=None, help="Max learning rate")
        parser.add_argument("--lr-min", type=float, default=None, help="Min learning rate")
        parser.add_argument("--t-warmup", type=int, default=None, help="Warmup iterations")
        parser.add_argument("--t-cos", type=int, default=None, help="Cosine decay iterations")

        # Optimizer
        parser.add_argument("--betas", type=str, default=None, help="AdamW betas")
        parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
        parser.add_argument("--max-grad", type=float, default=None, help="Max gradient norm")

        # Checkpointing
        parser.add_argument("--ckpt-iter", type=int, default=None, help="Checkpoint interval")
        parser.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint path")
        parser.add_argument("--from-ckpt", type=str, default=None, help="Resume from checkpoint")

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


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    print(config)
    