from src.callbacks.base import Callback


class ModelLogger(Callback):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def print_accelerate_info(self, accelerator):
        if self.verbose:
            print(f"Using Accelerator: {accelerator.state}")
            print(f"Device: {accelerator.device}")
    
    def print_model_info(self, model):
        if self.verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def on_step_end(self, accelerator, model, metrics=None):
        pass
    
    def on_epoch_end(self, accelerator, model):
        pass
    
    def on_training_end(self, accelerator, model):
        if self.verbose:
            print("Training completed!")
