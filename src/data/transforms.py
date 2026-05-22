import torch
from typing import Dict, Callable, List


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, item: Dict) -> Dict:
        for transform in self.transforms:
            item = transform(item)
        return item


class ToTensor:
    def __call__(self, item: Dict) -> Dict:
        result = {}
        for key, value in item.items():
            if isinstance(value, (int, float)):
                result[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, list):
                result[key] = torch.tensor(value)
            else:
                result[key] = value
        return result


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, item: Dict) -> Dict:
        result = item.copy()
        if 'input' in result:
            input_tensor = result['input']
            if isinstance(input_tensor, torch.Tensor) and input_tensor.dim() == 3:
                for t, m, s in zip(input_tensor, self.mean, self.std):
                    t.sub_(m).div_(s)
        return result
