import torch
import os
import json
from tqdm import tqdm
from datetime import datetime
from ..utils import Logger, Metrics, EarlyStopping


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = model.device
        
        self.logger = Logger(config)
        self.metrics = Metrics()
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 5),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train(self, train_loader, val_loader, epochs=10):
        self.logger.log(f"Training started with {epochs} epochs")
        
        for epoch in range(epochs):
            train_loss, train_metrics = self._train_epoch(train_loader)
            val_loss, val_metrics = self._val_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            self.logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            
            self._save_checkpoint(epoch)
            
            if self.early_stopping(val_loss):
                self.logger.log(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.logger.log(f"Training completed. Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return self.history

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for x, y in tqdm(loader, desc="Training"):
            loss, outputs = self.model.train_step(x, y)
            total_loss += loss
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        metrics = self.metrics.compute(all_labels, all_preds)
        return avg_loss, metrics

    def _val_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validating"):
                loss, outputs = self.model.eval_step(x, y)
                total_loss += loss
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        metrics = self.metrics.compute(all_labels, all_preds)
        return avg_loss, metrics

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Evaluating"):
                loss, outputs = self.model.eval_step(x, y)
                total_loss += loss
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        metrics = self.metrics.compute(all_labels, all_preds)
        
        self.logger.log(f"Test Loss: {avg_loss:.4f}")
        self.logger.log(f"Test Metrics: {metrics}")
        
        return avg_loss, metrics

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        
        self.model.save(path)
        
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def load_best_model(self):
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.model.load(best_model_path)
            self.logger.log("Loaded best model")
        else:
            self.logger.log("No best model found")