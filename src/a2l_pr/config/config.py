import yaml
from typing import Dict, Any

class ConfigManager:
    """Configuration manager for the perturbation package."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or {}

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def get_dataset_defaults(self, dataset_type: str) -> Dict[str, Any]:
        """Get default metadata configurations for a given dataset type."""
        defaults = self.config.get('datasets', {}).get(dataset_type, {})
        return defaults

    def get_perturbation_config(self) -> Dict[str, Any]:
        return self.config.get('perturbations', {})
