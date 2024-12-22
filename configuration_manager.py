import yaml
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from functools import reduce
import operator
import json

class ConfigurationManager:
    """
    Enhanced configuration manager with support for:
    - External logger injection
    - YAML configuration loading and saving
    - Environment variable overrides
    - GPT-specific validation rules
    - Nested setting access
    """
    
    # Default validation rules for GPT configuration
    GPT_VALIDATION_RULES = {
        "model_configs": {
            "vocab_size": lambda x: isinstance(x, int) and x > 0,
            "context_length": lambda x: isinstance(x, int) and x > 0,
            "emb_dim": lambda x: isinstance(x, int) and x > 0,
            "n_heads": lambda x: isinstance(x, int) and x > 0,
            "n_layers": lambda x: isinstance(x, int) and x > 0,
            "drop_rate": lambda x: isinstance(x, float) and 0 <= x <= 1,
            "qkv_bias": lambda x: isinstance(x, bool)
        },
        "training": {
            "train_ratio": lambda x: isinstance(x, float) and 0 < x < 1,
            "num_epochs": lambda x: isinstance(x, int) and x > 0,
            "batch_size": lambda x: isinstance(x, int) and x > 0,
            "subset_ratio": lambda x: isinstance(x, float) and 0 < x <= 1
        }
    }
    
    @staticmethod
    def create_default_logger(log_level: int = logging.INFO) -> logging.Logger:
        """
        Create a default logger if none is provided.
        
        Args:
            log_level (int): Logging level to use
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger("ConfigurationManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
        return logger
    
    def __init__(self, 
                 config_path: Union[str, Path], 
                 logger: Optional[logging.Logger] = None,
                 log_level: int = logging.INFO,
                 env_prefix: str = "GPT_"):
        """
        Initialize the ConfigurationManager.
        
        Args:
            config_path (Union[str, Path]): Path to YAML configuration file
            logger (Optional[logging.Logger]): External logger instance
            log_level (int): Logging level (used only if logger is not provided)
            env_prefix (str): Prefix for environment variables to override settings
        """
        self.config_path = Path(config_path)
        self.env_prefix = env_prefix
        self.logger = logger or self.create_default_logger(log_level)
        self.config = self._load_config()
        self._apply_environment_overrides()
            
    def _load_config(self) -> Dict:
        """Load and parse YAML configuration file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
                self.logger.info(f"Configuration loaded successfully from {self.config_path}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e}")
            raise
            
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                config_path = env_var[len(self.env_prefix):].lower().replace('_', '.')
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                    
                self.set_setting(config_path, parsed_value)
                self.logger.info(f"Override applied from environment: {env_var} -> {config_path}")
                
    def get_setting(self, path: str, default: Any = None) -> Any:
        """
        Get setting value using dot notation path.
        
        Args:
            path (str): Dot-separated path to setting
            default (Any): Default value if path doesn't exist
        """
        try:
            keys = path.split('.')
            return reduce(operator.getitem, keys, self.config)
        except (KeyError, TypeError):
            self.logger.warning(f"Setting not found: {path}, returning default: {default}")
            return default
            
    def set_setting(self, path: str, value: Any) -> None:
        """
        Set setting value using dot notation path.
        
        Args:
            path (str): Dot-separated path to setting
            value (Any): Value to set
        """
        keys = path.split('.')
        current = self.config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
        
    def save_config(self, output_path: Optional[Path] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path (Optional[Path]): Path to save to, defaults to original path
        """
        save_path = output_path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
            
    def validate_gpt_config(self) -> bool:
        """
        Validate GPT-specific configuration rules.
        
        Returns:
            bool: True if validation passes
        """
        validation_errors = []
        
        for section, rules in self.GPT_VALIDATION_RULES.items():
            section_config = self.get_setting(section)
            if not section_config:
                validation_errors.append(f"Missing required section: {section}")
                continue
                
            for param, validator in rules.items():
                value = section_config.get(param)
                if value is None:
                    validation_errors.append(f"Missing required parameter: {section}.{param}")
                elif not validator(value):
                    validation_errors.append(
                        f"Invalid value for {section}.{param}: {value}"
                    )
        
        if validation_errors:
            self.logger.error("GPT configuration validation failed:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            return False
        
        self.logger.info("GPT configuration validation passed")
        return True
        
    def print_settings(self, prefix: str = '') -> None:
        """Print all settings in hierarchical format."""
        def _print_dict(d: Dict, prefix: str = '') -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    self.logger.info(f"{prefix}{key}:")
                    _print_dict(value, prefix + '  ')
                else:
                    self.logger.info(f"{prefix}{key}: {value}")
                    
        self.logger.info("Current Configuration Settings:")
        _print_dict(self.config)

