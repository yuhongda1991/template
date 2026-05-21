# 完备 PyTorch 训练框架实现计划

**目标：** 构建一个模块化、可扩展、功能完备的 PyTorch 训练框架，包含数据加载、模型管理、训练循环、回调系统、日志记录、检查点管理等核心功能。

**架构设计：** 采用分层架构，将配置管理、数据处理、模型定义、训练逻辑、回调系统解耦。核心 Trainer 类负责训练流程编排，通过回调系统实现灵活扩展。

**技术栈：** PyTorch 2.0+, PyTorch Lightning 设计理念、YAML 配置、wandb 日志集成（可选）、tensorboard 支持

---

## 文件结构

```
/workspace/
├── configs/
│   └── default_config.yaml          # 默认配置文件
├── src/
│   ├── __init__.py
│   ├── config.py                     # 配置管理模块
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # 数据集基类
│   │   └── transforms.py             # 数据增强
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py                   # 模型基类
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # 核心训练器
│   │   ├── optimizer.py              # 优化器工厂
│   │   └── scheduler.py              # 学习率调度器
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── base.py                   # 回调基类
│   │   ├── checkpoint.py             # 模型检查点
│   │   ├── early_stopping.py         # 早停
│   │   └── logger.py                 # 日志回调
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── device.py                 # 设备管理
│   │   ├── metrics.py                # 评估指标
│   │   └── seed.py                   # 随机种子设置
│   └── cli.py                        # 命令行接口
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_trainer.py
│   └── test_callbacks.py
├── examples/
│   ├── mnist_example.py              # MNIST 示例
│   └── imagenet_example.py           # ImageNet 示例
├── requirements.txt
└── README.md
```

---

## 任务 1: 配置管理系统

**文件：**
- 创建: `src/config.py`
- 测试: `tests/test_config.py`

- [ ] **步骤 1: 编写配置类的单元测试**

```python
import pytest
from src.config import Config

def test_config_load_from_yaml(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
model:
  name: resnet18
  num_classes: 10
training:
  epochs: 100
  batch_size: 32
  lr: 0.001
    """)
    
    config = Config.from_yaml(str(config_file))
    assert config.model.name == "resnet18"
    assert config.training.epochs == 100
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_config.py::test_config_load_from_yaml -v`
预期: FAIL (Config 类未定义)

- [ ] **步骤 3: 实现配置管理类**

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml
from pathlib import Path

@dataclass
class Config:
    @staticmethod
    def from_yaml(path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return Config._from_dict(data)
    
    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> 'Config':
        model_data = data.get('model', {})
        training_data = data.get('training', {})
        
        return Config(
            model=ModelConfig(**model_data),
            training=TrainingConfig(**training_data)
        )

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
```

- [ ] **步骤 4: 运行测试确认通过**

运行: `pytest tests/test_config.py::test_config_load_from_yaml -v`
预期: PASS

- [ ] **步骤 5: 添加更多配置测试并提交**

---

## 任务 2: 工具模块 (设备、随机种子)

**文件：**
- 创建: `src/utils/device.py`
- 创建: `src/utils/seed.py`
- 创建: `src/utils/metrics.py`
- 创建: `src/utils/__init__.py`
- 测试: `tests/test_utils.py`

- [ ] **步骤 1: 编写设备管理测试**

```python
import torch
from src.utils.device import get_device, set_device

def test_get_device():
    device = get_device()
    assert device in ['cuda', 'cpu']
    if torch.cuda.is_available():
        assert device == 'cuda'
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_utils.py -v`
预期: FAIL (模块未定义)

- [ ] **步骤 3: 实现设备管理模块**

```python
import torch

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = get_device()
    return torch.device(device)

def to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    return data
```

- [ ] **步骤 4: 实现随机种子模块**

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

- [ ] **步骤 5: 实现指标计算模块**

```python
import torch
from typing import Dict

class Metrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
    
    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor):
        self.loss_sum += loss * targets.size(0)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        return {
            'accuracy': self.correct / self.total if self.total > 0 else 0.0,
            'loss': self.loss_sum / self.total if self.total > 0 else 0.0
        }
```

- [ ] **步骤 6: 运行测试确认通过并提交**

---

## 任务 3: 回调系统

**文件：**
- 创建: `src/callbacks/base.py`
- 创建: `src/callbacks/checkpoint.py`
- 创建: `src/callbacks/early_stopping.py`
- 创建: `src/callbacks/logger.py`
- 创建: `src/callbacks/__init__.py`
- 测试: `tests/test_callbacks.py`

- [ ] **步骤 1: 编写回调基类测试**

```python
from src.callbacks.base import Callback

def test_callback_interface():
    class TestCallback(Callback):
        def on_epoch_end(self, trainer, epoch, logs=None):
            logs['test_value'] = epoch * 2
    
    callback = TestCallback()
    assert hasattr(callback, 'on_epoch_end')
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_callbacks.py::test_callback_interface -v`
预期: FAIL

- [ ] **步骤 3: 实现回调基类**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class Callback(ABC):
    def on_train_begin(self, trainer, logs=None):
        pass
    
    def on_train_end(self, trainer, logs=None):
        pass
    
    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        pass
    
    def on_batch_begin(self, trainer, batch, logs=None):
        pass
    
    def on_batch_end(self, trainer, batch, logs=None):
        pass

class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
    
    def append(self, callback):
        self.callbacks.append(callback)
    
    def _invoke(self, method_name, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(*args, **kwargs)
```

- [ ] **步骤 4: 实现模型检查点回调**

```python
import torch
from pathlib import Path
from src.callbacks.base import Callback

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 save_last: bool = True, verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        should_save = False
        if self.save_best_only:
            if self.mode == 'min' and current_value < self.best_value:
                self.best_value = current_value
                should_save = True
            elif self.mode == 'max' and current_value > self.best_value:
                self.best_value = current_value
                should_save = True
        else:
            should_save = True
        
        if should_save:
            self._save_checkpoint(trainer, epoch, is_best=should_save)
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved, saved checkpoint")
    
    def _save_checkpoint(self, trainer, epoch, is_best=False):
        filepath = Path(self.filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_value': self.best_value
        }
        
        torch.save(checkpoint, filepath / 'last.pt')
        
        if is_best:
            torch.save(checkpoint, filepath / 'best.pt')
```

- [ ] **步骤 5: 实现早停回调**

```python
from src.callbacks.base import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 mode: str = 'min', min_delta: float = 0.0, verbose: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, logs=None):
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {epoch} epochs")
```

- [ ] **步骤 6: 运行测试确认通过并提交**

---

## 任务 4: 优化器和学习率调度器

**文件：**
- 创建: `src/training/optimizer.py`
- 创建: `src/training/scheduler.py`
- 测试: `tests/test_optimizer.py`

- [ ] **步骤 1: 编写优化器工厂测试**

```python
import torch.nn as nn
from src.training.optimizer import OptimizerFactory

def test_create_sgd():
    model = nn.Linear(10, 10)
    opt = OptimizerFactory.create(model, 'sgd', lr=0.01, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)

def test_create_adam():
    model = nn.Linear(10, 10)
    opt = OptimizerFactory.create(model, 'adam', lr=0.001)
    assert isinstance(opt, torch.optim.Adam)
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_optimizer.py -v`
预期: FAIL

- [ ] **步骤 3: 实现优化器工厂**

```python
import torch
from torch import nn

class OptimizerFactory:
    _optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop,
        'adagrad': torch.optim.Adagrad,
    }
    
    @classmethod
    def create(cls, model: nn.Module, name: str = 'sgd', **kwargs) -> torch.optim.Optimizer:
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {name}. Available: {list(cls._optimizers.keys())}")
        
        optimizer_cls = cls._optimizers[name]
        return optimizer_cls(model.parameters(), **kwargs)
    
    @classmethod
    def register(cls, name: str, optimizer_cls):
        cls._optimizers[name] = optimizer_cls
```

- [ ] **步骤 4: 实现学习率调度器工厂**

```python
import torch
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
    ExponentialLR, MultiStepLR
)

class SchedulerFactory:
    _schedulers = {
        'step': StepLR,
        'cosine': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'exponential': ExponentialLR,
        'multistep': MultiStepLR,
    }
    
    @classmethod
    def create(cls, optimizer, name: str = 'cosine', **kwargs):
        if name == 'plateau':
            return cls._schedulers[name](optimizer, **kwargs)
        return cls._schedulers[name](optimizer, **kwargs)
```

- [ ] **步骤 5: 运行测试确认通过并提交**

---

## 任务 5: 数据集基类

**文件：**
- 创建: `src/data/dataset.py`
- 创建: `src/data/transforms.py`
- 测试: `tests/test_data.py`

- [ ] **步骤 1: 编写数据集基类测试**

```python
from torch.utils.data import DataLoader
from src.data.dataset import BaseDataset

def test_base_dataset():
    class DummyDataset(BaseDataset):
        def __len__(self):
            return 100
        
        def _load_item(self, idx):
            return {'input': idx, 'label': idx % 10}
    
    dataset = DummyDataset()
    assert len(dataset) == 100
    item = dataset[0]
    assert 'input' in item and 'label' in item
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_data.py -v`
预期: FAIL

- [ ] **步骤 3: 实现数据集基类**

```python
from torch.utils.data import Dataset
from typing import Dict, Any, List
import torch

class BaseDataset(Dataset):
    def __init__(self, data_dir: str = None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data: List[Any] = []
        self._load_data()
    
    def _load_data(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self._load_item(idx)
        
        if self.transform:
            item = self.transform(item)
        
        return item

class ConcatDataset(BaseDataset):
    def __init__(self, datasets: List[BaseDataset]):
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()
    
    def _get_cumulative_sizes(self):
        sizes = []
        for dataset in self.datasets:
            sizes.append(len(dataset))
        return sizes
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def _load_item(self, idx):
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        else:
            dataset_idx = len(self.datasets) - 1
        
        return self.datasets[dataset_idx][idx]
```

- [ ] **步骤 4: 实现数据增强模块**

```python
import torch
from typing import Dict, Callable

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, item: Dict) -> Dict:
        for transform in self.transforms:
            item = transform(item)
        return item

class ToTensor:
    def __call__(self, item: Dict) -> Dict:
        result = {}
        for key, value in item.items():
            if isinstance(value, (int, float)):
                result[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, list):
                result[key] = torch.tensor(value)
            else:
                result[key] = value
        return result

class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, item: Dict) -> Dict:
        result = item.copy()
        if 'input' in result:
            input_tensor = result['input']
            if isinstance(input_tensor, torch.Tensor) and input_tensor.dim() == 3:
                for t, m, s in zip(input_tensor, self.mean, self.std):
                    t.sub_(m).div_(s)
        return result
```

- [ ] **步骤 5: 运行测试确认通过并提交**

---

## 任务 6: 模型基类

**文件：**
- 创建: `src/models/base.py`
- 测试: `tests/test_models.py`

- [ ] **步骤 1: 编写模型基类测试**

```python
import torch
from src.models.base import BaseModel

def test_base_model():
    class SimpleModel(BaseModel):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
        
        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())
    
    model = SimpleModel(10, 5)
    assert model.get_num_params() > 0
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_models.py -v`
预期: FAIL

- [ ] **步骤 3: 实现模型基类**

```python
import torch
from torch import nn
from typing import Optional, Dict, Any

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def summary(self) -> Dict[str, Any]:
        return {
            'num_params': self.get_num_params(),
            'trainable_params': self.get_trainable_params(),
            'layers': list(self.named_children())
        }
```

- [ ] **步骤 4: 运行测试确认通过并提交**

---

## 任务 7: 核心训练器

**文件：**
- 创建: `src/training/trainer.py`
- 测试: `tests/test_trainer.py`

- [ ] **步骤 1: 编写训练器单元测试**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory

def test_trainer_init():
    model = nn.Linear(10, 2)
    optimizer = OptimizerFactory.create(model, 'sgd', lr=0.01)
    trainer = Trainer(model, optimizer, device='cpu')
    assert trainer.model is not None
    assert trainer.optimizer is not None
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_trainer.py::test_trainer_init -v`
预期: FAIL

- [ ] **步骤 3: 实现核心训练器类**

```python
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time

from src.callbacks.base import CallbackList
from src.utils.device import to_device

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        callbacks: Optional[list] = None,
        verbose: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.callbacks = CallbackList(callbacks)
        self.verbose = verbose
        
        self.model.to(self.device)
        self.current_epoch = 0
        self.global_step = 0
        
        self.callbacks._invoke('on_train_begin', self)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 10) -> Dict[str, Any]:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.callbacks._invoke('on_epoch_begin', self, epoch)
            
            train_metrics = self._train_epoch(train_loader)
            history['train_loss'].extend(train_metrics['loss'])
            
            logs = {'epoch': epoch, **train_metrics}
            
            if val_loader:
                val_metrics = self._validate(val_loader)
                history['val_loss'].extend(val_metrics['loss'])
                logs['val_loss'] = val_metrics['loss']
                logs['val_acc'] = val_metrics['accuracy']
            
            self.callbacks._invoke('on_epoch_end', self, epoch, logs)
            
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            elif self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(logs.get('val_loss', 0))
        
        self.callbacks._invoke('on_train_end', self, logs)
        return history
    
    def _train_epoch(self, loader: DataLoader) -> Dict[str, list]:
        self.model.train()
        losses = []
        
        for batch_idx, batch in enumerate(loader):
            self.callbacks._invoke('on_batch_begin', self, batch_idx)
            
            data, target = batch
            data, target = to_device(data, self.device), to_device(target, self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            self.global_step += 1
            
            self.callbacks._invoke('on_batch_end', self, batch_idx, {'loss': loss.item()})
        
        return {'loss': losses}
    
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = to_device(data, self.device), to_device(target, self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * target.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    def predict(self, loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                data = batch[0] if isinstance(batch, tuple) else batch
                data = to_device(data, self.device)
                output = self.model(data)
                predictions.append(output)
        
        return torch.cat(predictions, dim=0)
```

- [ ] **步骤 4: 运行测试确认通过并提交**

---

## 任务 8: 命令行接口

**文件：**
- 创建: `src/cli.py`
- 测试: `tests/test_cli.py`

- [ ] **步骤 1: 编写 CLI 测试**

```python
from src.cli import parse_args

def test_parse_args():
    args = parse_args(['--config', 'configs/default_config.yaml', '--epochs', '50'])
    assert args.config == 'configs/default_config.yaml'
    assert args.epochs == 50
```

- [ ] **步骤 2: 运行测试确认失败**

运行: `pytest tests/test_cli.py -v`
预期: FAIL

- [ ] **步骤 3: 实现命令行接口**

```python
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Framework')
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    from src.config import Config
    from src.utils.seed import set_seed
    from src.training.trainer import Trainer
    from src.training.optimizer import OptimizerFactory
    from src.callbacks.checkpoint import ModelCheckpoint
    from src.callbacks.early_stopping import EarlyStopping
    
    set_seed(args.seed)
    
    config = Config.from_yaml(args.config)
    
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    
    print(f"Starting training with config: {config}")
    print(f"Training for {config.training.epochs} epochs")

if __name__ == '__main__':
    main()
```

- [ ] **步骤 4: 运行测试确认通过并提交**

---

## 任务 9: 示例代码

**文件：**
- 创建: `examples/mnist_example.py`
- 创建: `examples/imagenet_example.py`

- [ ] **步骤 1: 编写 MNIST 示例**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize

from src.config import Config
from src.models.base import BaseModel
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
from src.utils.seed import set_seed

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    set_seed(42)
    
    train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    val_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MnistModel()
    optimizer = OptimizerFactory.create(model, 'adam', lr=0.001)
    
    callbacks = [
        ModelCheckpoint('outputs/checkpoints', monitor='val_acc', mode='max', verbose=True),
        EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=True)
    ]
    
    trainer = Trainer(model, optimizer, callbacks=callbacks, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    history = trainer.train(train_loader, val_loader, epochs=10)
    print("Training completed!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")

if __name__ == '__main__':
    main()
```

- [ ] **步骤 2: 提交示例代码**

---

## 任务 10: 配置文件和依赖

**文件：**
- 创建: `configs/default_config.yaml`
- 创建: `requirements.txt`
- 创建: `tests/__init__.py`
- 创建: `src/__init__.py`

- [ ] **步骤 1: 创建默认配置文件**

```yaml
model:
  name: resnet18
  num_classes: 1000
  pretrained: false

training:
  epochs: 100
  batch_size: 32
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  optimizer: adam
  
scheduler:
  name: cosine
  min_lr: 0.00001

data:
  data_dir: ./data
  num_workers: 4
  pin_memory: true
  
logging:
  log_dir: ./logs
  save_checkpoint: true
  checkpoint_dir: ./checkpoints
  tensorboard: true
  wandb: false
```

- [ ] **步骤 2: 创建 requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
pyyaml>=6.0
tensorboard>=2.13.0
wandb>=0.15.0
pytest>=7.4.0
tqdm>=4.65.0
```

- [ ] **步骤 3: 创建初始化文件并提交所有文件**

---

## 任务 11: 最终集成测试

**文件：**
- 创建: `tests/test_integration.py`

- [ ] **步骤 1: 编写集成测试**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models.base import BaseModel
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
from src.utils.seed import set_seed

class DummyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

def test_full_training_pipeline():
    set_seed(42)
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=16)
    
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    model = DummyModel()
    optimizer = OptimizerFactory.create(model, 'sgd', lr=0.01)
    
    callbacks = [
        ModelCheckpoint('test_checkpoints', monitor='val_acc', mode='max', verbose=False),
        EarlyStopping(monitor='val_acc', patience=3, mode='max', verbose=False)
    ]
    
    trainer = Trainer(model, optimizer, callbacks=callbacks, device='cpu')
    history = trainer.train(train_loader, val_loader, epochs=5)
    
    assert len(history['train_loss']) > 0
    print("Integration test passed!")
```

- [ ] **步骤 2: 运行完整测试套件**

运行: `pytest tests/ -v`
预期: 所有测试通过

- [ ] **步骤 3: 最终提交**
