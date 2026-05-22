import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time
from tqdm import tqdm

from src.callbacks.base import CallbackList
from src.utils.device import to_device
from src.config import Config


class Profiler:
    def __init__(self, enabled: bool = False, print_steps: int = 10):
        self.enabled = enabled
        self.print_steps = print_steps
        self.step_count = 0
        self.timings = {}
    
    def __enter__(self, name: str):
        if self.enabled:
            self.current_name = name
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            elapsed = time.perf_counter() - self.start_time
            if self.current_name not in self.timings:
                self.timings[self.current_name] = []
            self.timings[self.current_name].append(elapsed)
    
    def __call__(self, name: str):
        class ProfilerContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
            
            def __enter__(self):
                if self.profiler.enabled:
                    self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                if self.profiler.enabled:
                    elapsed = time.perf_counter() - self.start_time
                    if self.name not in self.profiler.timings:
                        self.profiler.timings[self.name] = []
                    self.profiler.timings[self.name].append(elapsed)
        
        return ProfilerContext(self, name)
    
    def record(self, name: str, elapsed: float):
        if self.enabled:
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
    
    def step(self):
        self.step_count += 1
        if self.enabled and self.step_count % self.print_steps == 0:
            self.print_stats()
    
    def print_stats(self):
        if not self.enabled:
            return
        print("\nProfiler Stats:")
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            print(f"  {name}: {avg_time*1000:.2f}ms (avg over {len(times)} steps)")
        print()


def initialize_deepspeed_gradient_checkpointing(accelerator):
    pass


def launch_batch_training_task(
    accelerator,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    model_logger,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Config,
):
    """
    Optimized version of the batch training task launcher.
    
    Key optimizations:
    1. Fixed model device placement order (prepare first, then device management)
    2. Optimized gradient zeroing placement
    3. Improved code structure and readability
    4. Better pure dataloader mode handling
    """
    profiler = Profiler(enabled=config.training.enable_profile, 
                       print_steps=config.training.profile_steps)
    
    model_logger.print_accelerate_info(accelerator)
    
    # Prepare everything in the correct order
    if not config.training.pure_dataloader:
        model_logger.print_model_info(model)
        # Use accelerator.prepare() for all components - it handles device placement
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        initialize_deepspeed_gradient_checkpointing(accelerator)
    else:
        dataloader = accelerator.prepare(dataloader)
    
    num_iters = len(dataloader)
    
    # Training loop
    for epoch_id in range(config.training.num_epochs):
        data_iter = iter(dataloader)
        
        for _ in tqdm(range(num_iters), desc=f"Epoch {epoch_id}/{config.training.num_epochs}"):
            t0 = time.perf_counter()
            
            # Data loading phase
            with profiler("data"):
                data = next(data_iter)
            
            # Skip training if in pure dataloader mode
            if config.training.pure_dataloader:
                profiler.step()
                continue
            
            # Zero gradients at the beginning of the step (better practice)
            optimizer.zero_grad()
            
            # Forward and backward pass with gradient accumulation
            with accelerator.accumulate(model):
                with profiler("forward"):
                    output = model(data)
                    loss = loss_fn(output, data)
                
                with profiler("backward"):
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
            
            # Log metrics
            metrics = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            model_logger.on_step_end(accelerator, model, metrics=metrics)
            
            profiler.record("total", time.perf_counter() - t0)
            profiler.step()
        
        model_logger.on_epoch_end(accelerator, model)
    
    model_logger.on_training_end(accelerator, model)


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
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
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
