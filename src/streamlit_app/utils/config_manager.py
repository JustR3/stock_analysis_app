import os
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)
            
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load and merge configuration files."""
        # Load base config
        base_config_path = self.config_dir / "config.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Load environment-specific config
        env = os.getenv("APP_ENV", "dev")
        env_config_path = self.config_dir / f"{env}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
                self._merge_configs(self.config, env_config)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy() 