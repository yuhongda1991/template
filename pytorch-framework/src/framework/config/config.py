import yaml
import os


class Config:
    def __init__(self, config_dict=None, config_path=None):
        self._config = {}
        
        if config_path is not None:
            self.load_from_file(config_path)
        
        if config_dict is not None:
            self._config.update(config_dict)
        
        self._set_defaults()

    def _set_defaults(self):
        defaults = {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss_fn': 'cross_entropy',
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.001,
            'use_wandb': False,
            'project_name': 'pytorch-framework',
            'experiment_name': 'default'
        }
        
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value

    def load_from_file(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._config.update(config_dict)
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    def save_to_file(self, path):
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value

    def __contains__(self, key):
        return key in self._config

    def __iter__(self):
        return iter(self._config)

    def keys(self):
        return self._config.keys()

    def values(self):
        return self._config.values()

    def items(self):
        return self._config.items()

    def __str__(self):
        return str(self._config)

    def __repr__(self):
        return repr(self._config)