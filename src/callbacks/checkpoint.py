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
