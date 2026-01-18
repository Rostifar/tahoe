import yaml
import torch
import argparse
import torch.nn as nn
from typing import Literal
from pydantic import BaseModel


class SchedulerConfig(BaseModel):
    lr_max: float
    lr_min: float
    t_warmup: int
    t_cos: int


class AdamWConfig(BaseModel):
    lr: float
    betas: tuple[float, float] = (0.99, 0.999)
    weight_decay: float = 1e-2

    def build(self, params) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )


class ExperimentConfig(BaseModel):
    name: str
    batch_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    theta: float

    device: Literal["cpu", "mps", "cuda"] | list[int]
    dtype: Literal["fp16", "fp32"]

    scheduler: SchedulerConfig | None = None
    optimizer: AdamWConfig

    max_grad: float = 1.0

    ckpt_iter: int
    ckpt_path: str
    val_iter: int | None = None
    max_iter: int | None = None
    from_ckpt: str | None = None

    @property
    def is_multi_gpu(self) -> bool:
        return isinstance(self.device, list)

    @property
    def device_ids(self) -> list[int]:
        if isinstance(self.device, list):
            return self.device
        return []

    @property
    def primary_device(self) -> torch.device:
        if isinstance(self.device, list):
            return torch.device(f"cuda:{self.device[0]}")
        return torch.device(self.device)

    def get_dtype(self) -> torch.dtype:
        return torch.float16 if self.dtype == "fp16" else torch.float32

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.primary_device)
        if self.is_multi_gpu:
            return nn.DataParallel(model, device_ids=self.device_ids)
        return model


class DataConfig(BaseModel):
    vocab: str
    train_set: str
    val_set: str


class Config(BaseModel):
    data: DataConfig
    experiments: dict[str, ExperimentConfig]

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        for name, exp in data.get("experiments", {}).items():
            exp["name"] = name
            if "optimizer" in exp and "betas" in exp["optimizer"]:
                if isinstance(exp["optimizer"]["betas"], list):
                    exp["optimizer"]["betas"] = tuple(exp["optimizer"]["betas"])
        return cls(**data)


    @classmethod
    def from_args(cls, args: list[str] | None = None) -> tuple[DataConfig, ExperimentConfig | None]:
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--config", type=str, required=True, help="Path to config file")
        parser.add_argument("--experiment", type=str, help="Experiment name (optional)")
        parsed = parser.parse_args(args)
        
        config = cls.from_yaml(parsed.config)
        experiment = config.experiments.get(parsed.experiment) if parsed.experiment else None
        return config.data, experiment


    def to_yaml(self, path: str) -> None:
        data = self.model_dump()
        for exp in data.get("experiments", {}).values():
            if "optimizer" in exp and "betas" in exp["optimizer"]:
                if isinstance(exp["optimizer"]["betas"], tuple):
                    exp["optimizer"]["betas"] = list(exp["optimizer"]["betas"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
