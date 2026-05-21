import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# 配置
config = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
    'mixed_precision': True,
    'distributed': True,
    'world_size': 2,  # GPU数量
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


def train(rank, world_size):
    """训练函数"""
    # 设置随机种子
    set_seed(config['seed'])
    
    # 分布式设置
    if config['distributed']:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据
    dataset = create_dummy_data()
    
    # 分布式数据加载
    if config['distributed']:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank
        )
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=True
        )
    
    # 创建模型
    model = SimpleModel().to(device)
    
    if config['distributed']:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if config['mixed_precision'] else None
    
    # 训练循环
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
                # 混合精度前向传播
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 正常精度训练
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # 打印日志（仅在主进程）
        if not config['distributed'] or rank == 0:
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{config["epochs"]}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 清理分布式进程组
    if config['distributed']:
        dist.destroy_process_group()


def main():
    """主函数"""
    if config['distributed']:
        # 启动多进程分布式训练
        mp.spawn(
            train,
            args=(config['world_size'],),
            nprocs=config['world_size'],
            join=True
        )
    else:
        # 单进程训练
        train(0, 1)


if __name__ == '__main__':
    main()
