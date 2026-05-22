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
    
    def _load_item(self, idx):
        raise NotImplementedError


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
