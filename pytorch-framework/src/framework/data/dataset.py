import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

    @classmethod
    def from_numpy(cls, X, y=None, transform=None):
        data = torch.from_numpy(X).float()
        if y is not None:
            labels = torch.from_numpy(y).long()
        else:
            labels = None
        return cls(data, labels, transform)

    @classmethod
    def from_tensor(cls, X, y=None, transform=None):
        return cls(X, y, transform)