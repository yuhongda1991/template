import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload


config = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
    'mixed_precision': True,
    'distributed': True,
    'use_fsdp': True,
    'fsdp_sharding_strategy': 'full_shard',
    'fsdp_cpu_offload': False,
    'world_size': 2,
    'seed': 42
}


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleModel(nn.Module):
    """简单的分类模型"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data():
    """创建虚拟数据"""
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    return TensorDataset(X, y)


def setup_fsdp_model(model, rank, mixed_precision=False, cpu_offload=False, sharding_strategy='full_shard'):
    """创建FSDP模型包装器"""
    if mixed_precision:
        mp_dtype = torch.float16
        mixed_precision_cfg = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision_cfg = None
        mp_dtype = None
    
    cpu_offload_cfg = CPUOffload(offload_params=False) if cpu_offload else None
    
    sharding_strategy_map = {
        'full_shard': ShardingStrategy.FULL_SHARD,
        'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
        'no_shard': ShardingStrategy.NO_SHARD,
        'hybrid_shard': ShardingStrategy.HYBRID_SHARD,
    }
    fsdp_sharding = sharding_strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    model = FSDP(
        model,
        sharding_strategy=fsdp_sharding,
        mixed_precision=mixed_precision_cfg,
        cpu_offload=cpu_offload_cfg,
        device_ids=[rank] if torch.cuda.is_available() else None,
        bucket_cap_mb=25,
        use_orig_params=True,
    )
    
    return model


def train(rank, world_size):
    """训练函数"""
    set_seed(config['seed'])
    
    if config['distributed']:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = create_dummy_data()
    
    if config['distributed']:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], sampler=sampler,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=True
        )
    
    model = SimpleModel()
    
    if config['distributed']:
        if config['use_fsdp']:
            model = setup_fsdp_model(
                model, rank,
                mixed_precision=config['mixed_precision'],
                cpu_offload=config['fsdp_cpu_offload'],
                sharding_strategy=config['fsdp_sharding_strategy']
            )
        else:
            model = model.to(device)
            model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    else:
        model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    if config['mixed_precision'] and config['use_fsdp']:
        scaler = ShardedGradScaler() if torch.cuda.is_available() else None
    elif config['mixed_precision']:
        scaler = GradScaler() if torch.cuda.is_available() else None
    else:
        scaler = None
    
    model.train()
    for epoch in range(config['epochs']):
        if config['distributed']:
            sampler.set_epoch(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if config['mixed_precision'] and scaler is not None:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        if not config['distributed'] or rank == 0:
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{config["epochs"]}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    if config['distributed'] and config['use_fsdp']:
        model.barrier()
    
    if config['distributed']:
        dist.destroy_process_group()


def main():
    """主函数"""
    if config['distributed']:
        mp.spawn(
            train,
            args=(config['world_size'],),
            nprocs=config['world_size'],
            join=True
        )
    else:
        train(0, 1)


if __name__ == '__main__':
    main()
