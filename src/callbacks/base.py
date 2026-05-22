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
