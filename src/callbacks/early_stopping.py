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
