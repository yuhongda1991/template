# PyTorch Training Framework

A modular, extensible PyTorch training framework with optimized training functions.

## Features

- **Configurable Training**: Easy configuration via YAML files
- **Callback System**: Flexible callbacks for checkpointing, early stopping, and logging
- **Optimizer & Scheduler Factories**: Easy creation of optimizers and learning rate schedulers
- **Optimized Training Function**: Enhanced `launch_batch_training_task` with improved code structure and efficiency

## Key Optimization: launch_batch_training_task

The `launch_batch_training_task` function has been optimized with several improvements:

1. **Fixed Device Placement Order**: Properly uses `accelerator.prepare()` for all components before training
2. **Optimized Gradient Zeroing**: Zeroes gradients at the beginning of each step (best practice)
3. **Improved Code Structure**: Better readability and maintainability
4. **Enhanced Profiler**: A fully functional profiler class for performance monitoring
5. **Better Pure Dataloader Mode**: Cleaner handling of the pure dataloader testing mode

## Project Structure

```
/workspace/
├── configs/                  # Configuration files
│   └── default_config.yaml
├── src/
│   ├── config.py            # Configuration management
│   ├── data/                # Data handling
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/              # Model definitions
│   │   └── base.py
│   ├── training/            # Training logic
│   │   ├── trainer.py       # Core Trainer and launch_batch_training_task
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── callbacks/           # Callbacks
│   │   ├── base.py
│   │   ├── checkpoint.py
│   │   ├── early_stopping.py
│   │   └── logger.py
│   └── utils/               # Utilities
│       ├── device.py
│       ├── seed.py
│       └── metrics.py
├── examples/                # Example scripts
│   └── mnist_example.py
├── tests/                   # Tests
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage with Trainer Class

```python
from src.training.trainer import Trainer
from src.training.optimizer import OptimizerFactory

# Create model, optimizer
model = YourModel()
optimizer = OptimizerFactory.create(model, 'adam', lr=0.001)

# Create trainer
trainer = Trainer(model, optimizer)

# Train
history = trainer.train(train_loader, val_loader, epochs=10)
```

### Using the Optimized launch_batch_training_task

```python
from src.training.trainer import launch_batch_training_task
from src.config import Config
from src.callbacks.logger import ModelLogger

# Load config
config = Config.from_yaml('configs/default_config.yaml')

# Create components
model = YourModel()
optimizer = ...
scheduler = ...
loss_fn = ...
model_logger = ModelLogger()

# Launch training
launch_batch_training_task(
    accelerator=accelerator,
    dataloader=dataloader,
    model=model,
    model_logger=model_logger,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)
```

## License

MIT
