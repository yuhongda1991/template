import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import random_split


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader = self._create_dataloader()

    def _create_dataloader(self):
        return TorchDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    @classmethod
    def split_dataset(cls, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        total = len(dataset)
        train_size = int(train_ratio * total)
        val_size = int(val_ratio * total)
        test_size = total - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def create_loaders(cls, dataset, batch_size=32, train_ratio=0.8, val_ratio=0.1, **kwargs):
        train_dataset, val_dataset, test_dataset = cls.split_dataset(dataset, train_ratio, val_ratio)
        
        train_loader = cls(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = cls(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_loader = cls(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        
        return train_loader, val_loader, test_loader