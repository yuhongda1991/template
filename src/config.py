from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 1000
    pretrained: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    enable_profile: bool = False
    profile_steps: int = 10
    pure_dataloader: bool = False
    num_epochs: int = 100


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    min_lr: float = 1e-5


@dataclass
class DataConfig:
    data_dir: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    save_checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"
    tensorboard: bool = True
    wandb: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_yaml(path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config._from_dict(data)
    
    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> 'Config':
        model_data = data.get('model', {})
        training_data = data.get('training', {})
        scheduler_data = data.get('scheduler', {})
        data_data = data.get('data', {})
        logging_data = data.get('logging', {})
        
        return Config(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data),
            scheduler=SchedulerConfig(**scheduler_data),
            data=DataConfig(**data_data),
            logging=LoggingConfig(**logging_data)
        )
