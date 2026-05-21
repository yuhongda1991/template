import os
import logging
from datetime import datetime


class Logger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.get('log_dir', 'logs')
        self.use_wandb = config.get('use_wandb', False)
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = self._create_logger()
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(
                    project=config.get('project_name', 'pytorch-framework'),
                    name=config.get('experiment_name', 'default'),
                    config=config._config
                )
            except ImportError:
                self.use_wandb = False
                self.log("wandb not installed, skipping wandb logging")

    def _create_logger(self):
        logger = logging.getLogger('framework')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(
            os.path.join(self.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def log(self, message):
        self.logger.info(message)

    def log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        message = (f"Epoch {epoch}: "
                   f"Train Loss={train_loss:.4f}, "
                   f"Val Loss={val_loss:.4f}, "
                   f"Train Metrics={train_metrics}, "
                   f"Val Metrics={val_metrics}")
        self.log(message)
        
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            log_dict.update({f'train_{k}': v for k, v in train_metrics.items()})
            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
            self.wandb.log(log_dict)

    def log_hyperparams(self, params):
        if self.use_wandb:
            self.wandb.config.update(params)

    def close(self):
        if self.use_wandb:
            self.wandb.finish()