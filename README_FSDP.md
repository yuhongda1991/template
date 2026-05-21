# PyTorch FSDP 训练框架使用说明

## 功能特性

本项目实现了一个支持 **FSDP (Fully Sharded Data Parallel)** 的 PyTorch 训练框架，具备以下特性：

- ✅ **原生 PyTorch**：直接使用 PyTorch 官方 API，无过度封装
- ✅ **FSDP 分布式训练**：支持模型参数分片，大幅降低显存占用
- ✅ **混合精度训练**：支持 FP16/BF16 混合精度训练
- ✅ **多种分片策略**：FULL_SHARD、SHARD_GRAD_OP、HYBRID_SHARD 等
- ✅ **CPU Offload**：可选将参数卸载到 CPU，进一步节省显存
- ✅ **单文件实现**：所有功能在一个 Python 文件中，简单易用

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
```

### 2. 基本使用

```bash
python train.py
```

## 配置说明

通过修改 `config` 字典来调整训练参数：

```python
config = {
    'batch_size': 64,              # 批次大小
    'epochs': 10,                  # 训练轮数
    'lr': 0.001,                  # 学习率
    'mixed_precision': True,      # 是否启用混合精度
    'distributed': True,          # 是否启用分布式训练
    'use_fsdp': True,             # 是否使用 FSDP
    'fsdp_sharding_strategy': 'full_shard',  # 分片策略
    'fsdp_cpu_offload': False,    # 是否启用 CPU Offload
    'world_size': 2,              # GPU 数量
    'seed': 42                    # 随机种子
}
```

## FSDP 分片策略

### 1. FULL_SHARD（推荐）
- 在前向和反向传播时都将参数分片
- 显存占用最低
- 通信开销中等
- 适合大模型训练

```python
'fsdp_sharding_strategy': 'full_shard'
```

### 2. SHARD_GRAD_OP
- 仅在反向传播时分片参数和梯度
- 显存占用中等
- 通信开销较低
- 适合中等规模模型

```python
'fsdp_sharding_strategy': 'shard_grad_op'
```

### 3. HYBRID_SHARD
- 结合数据并行和模型并行
- 在本地和跨节点间分片参数
- 适合多节点训练

```python
'fsdp_sharding_strategy': 'hybrid_shard'
```

### 4. NO_SHARD
- 不分片，仅使用 DDP
- 显存占用最高
- 通信开销最低

```python
'fsdp_sharding_strategy': 'no_shard'
```

## 使用场景示例

### 单 GPU 训练（不使用分布式）
```python
config['distributed'] = False
config['use_fsdp'] = False
```

### 单 GPU 混合精度训练
```python
config['distributed'] = False
config['use_fsdp'] = False
config['mixed_precision'] = True
```

### 多 GPU FSDP 训练（推荐）
```python
config['distributed'] = True
config['use_fsdp'] = True
config['mixed_precision'] = True
config['fsdp_sharding_strategy'] = 'full_shard'
config['world_size'] = 4  # 使用 4 张 GPU
```

### 超大模型（使用 CPU Offload）
```python
config['distributed'] = True
config['use_fsdp'] = True
config['mixed_precision'] = True
config['fsdp_sharding_strategy'] = 'full_shard'
config['fsdp_cpu_offload'] = True  # 将部分参数卸载到 CPU
config['world_size'] = 2
```

## 模型和数据替换

替换为自己的模型：
```python
class YourModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 定义你的模型结构
        pass
    
    def forward(self, x):
        # 定义前向传播
        return x
```

替换为自己的数据集：
```python
def create_dummy_data():
    # 替换为你的数据集
    from torch.utils.data import DataLoader, TensorDataset
    # dataset = YourDataset(...)
    return dataset
```

## 注意事项

1. **FSDP 需要 CUDA 环境**：FSDP 需要 CUDA 支持，建议使用 GPU 进行训练
2. **通信后端**：默认使用 NCCL 后端进行分布式训练
3. **多进程启动**：使用 `mp.spawn` 自动启动多进程
4. **混合精度兼容性**：FSDP 与 `ShardedGradScaler` 配合使用效果最佳

## 性能对比

| 策略 | 显存占用 | 通信开销 | 适用场景 |
|------|---------|---------|---------|
| DDP | 高 | 低 | 小模型 |
| FSDP (FULL_SHARD) | 最低 | 中等 | 大模型 |
| FSDP (SHARD_GRAD_OP) | 中等 | 低 | 中等模型 |
| FSDP + CPU Offload | 最低 | 高 | 超大模型 |

## 故障排除

### 问题：CUDA 不可用
**解决方案**：确保安装了 CUDA 版本的 PyTorch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 问题：多进程启动失败
**解决方案**：检查 `world_size` 是否与实际 GPU 数量匹配

### 问题：显存不足
**解决方案**：
1. 减小 `batch_size`
2. 使用 `FULL_SHARD` 策略
3. 启用 `fsdp_cpu_offload`
4. 启用混合精度 `mixed_precision: True`
