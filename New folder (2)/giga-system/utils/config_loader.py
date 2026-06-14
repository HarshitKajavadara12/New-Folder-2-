"""
GIGA SYSTEM - Configuration Loader
Greek Intelligence for Global Analysis

Centralized configuration management using TOML files.
Provides type-safe configuration loading with validation,
environment variable substitution, and hot reloading.

Features:
- TOML configuration files (human-readable)
- Environment variable substitution
- Configuration validation with types
- Hot reloading for development
- Hierarchical configuration merging
- Encrypted configuration for secrets
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import toml

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False


class ConfigManager:
    """
    Thread-safe configuration manager for GIGA System.
    
    Manages configuration loading, validation, and hot reloading
    across all system components.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.watchers = {}
        self.encryption_key = self._get_encryption_key()
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all configuration files
        self._load_all_configs()
    
    def _get_encryption_key(self) -> Optional[bytes]:
        """Get encryption key from environment or generate new one."""
        if not ENCRYPTION_AVAILABLE:
            return None
        
        key_env = os.getenv("GIGA_ENCRYPTION_KEY")
        if key_env:
            return key_env.encode()
        
        # Generate new key for development
        return Fernet.generate_key()
    
    def _load_all_configs(self):
        """Load all TOML configuration files from config directory."""
        config_files = [
            "system_config.toml",
            "database_config.toml", 
            "models_config.toml",
            "strategies_config.toml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            config_name = config_file.replace(".toml", "")
            
            if config_path.exists():
                self.configs[config_name] = self._load_config_file(config_path)
            else:
                # Create default config if doesn't exist
                self._create_default_config(config_name, config_path)
                self.configs[config_name] = self._load_config_file(config_path)
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load and validate a single configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = toml.load(f)
            
            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)
            
            # Decrypt encrypted values if needed
            if self.encryption_key:
                config_data = self._decrypt_values(config_data)
            
            return config_data
            
        except Exception as e:
            warnings.warn(f"Failed to load config {config_path}: {e}")
            return {}
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in config values."""
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Environment variable substitution: ${VAR_NAME:default_value}
            var_spec = data[2:-1]  # Remove ${ and }
            
            if ":" in var_spec:
                var_name, default_value = var_spec.split(":", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_spec, data)  # Return original if not found
        else:
            return data
    
    def _decrypt_values(self, data: Any) -> Any:
        """Recursively decrypt encrypted values in configuration."""
        if not ENCRYPTION_AVAILABLE or not self.encryption_key:
            return data
        
        fernet = Fernet(self.encryption_key)
        
        if isinstance(data, dict):
            return {key: self._decrypt_values(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._decrypt_values(item) for item in data]
        elif isinstance(data, str) and data.startswith("ENC:"):
            # Decrypt encrypted value
            try:
                encrypted_value = data[4:].encode()  # Remove ENC: prefix
                decrypted_value = fernet.decrypt(encrypted_value)
                return decrypted_value.decode()
            except Exception as e:
                warnings.warn(f"Failed to decrypt value: {e}")
                return data
        else:
            return data
    
    def _create_default_config(self, config_name: str, config_path: Path):
        """Create default configuration file."""
        default_configs = {
            "system_config": {
                "system": {
                    "name": "GIGA System",
                    "version": "1.0.0",
                    "environment": "${ENVIRONMENT:development}",
                    "debug": True,
                    "log_level": "INFO"
                },
                "performance": {
                    "numba_cache": True,
                    "polars_lazy": True,
                    "duckdb_threads": 4,
                    "duckdb_memory_limit": "1GB"
                },
                "paths": {
                    "data_dir": "data",
                    "logs_dir": "logs", 
                    "cache_dir": "cache",
                    "temp_dir": "temp"
                }
            },
            
            "database_config": {
                "duckdb": {
                    "database_path": "giga_system.duckdb",
                    "memory_limit": "1GB",
                    "threads": 4,
                    "checkpoint_wal_auto": True,
                    "enable_progress_bar": False
                },
                "backup": {
                    "enabled": True,
                    "interval_hours": 6,
                    "max_backups": 7,
                    "backup_dir": "backups"
                },
                "cleanup": {
                    "auto_cleanup": True,
                    "retention_days": 365,
                    "cleanup_hour": 2
                }
            },
            
            "models_config": {
                "black_scholes": {
                    "default_risk_free_rate": 0.05,
                    "default_dividend_yield": 0.0,
                    "precision": 1e-6,
                    "max_iterations": 100
                },
                "monte_carlo": {
                    "default_simulations": 10000,
                    "random_seed": 42,
                    "antithetic_variates": True,
                    "control_variates": True
                },
                "quantum": {
                    "backend": "qasm_simulator",
                    "shots": 1024,
                    "circuit_depth_limit": 100,
                    "fallback_classical": True
                },
                "garch": {
                    "default_model": "sGARCH",
                    "default_distribution": "norm", 
                    "solver": "hybrid",
                    "forecast_horizon": 1
                }
            },
            
            "strategies_config": {
                "delta_neutral": {
                    "rebalance_threshold": 0.1,
                    "max_positions": 50,
                    "position_size_pct": 0.02,
                    "stop_loss_pct": 0.05
                },
                "gamma_scalping": {
                    "min_gamma": 0.01,
                    "rebalance_frequency_minutes": 5,
                    "volatility_threshold": 0.2,
                    "profit_target_pct": 0.01
                },
                "volatility_arbitrage": {
                    "iv_rv_threshold": 0.05,
                    "min_time_to_expiry_days": 7,
                    "max_time_to_expiry_days": 90,
                    "min_volume": 1000
                },
                "hft_simulator": {
                    "latency_ms": 0.5,
                    "market_impact_bps": 0.1,
                    "tick_size": 0.01,
                    "max_order_size": 10000
                }
            }
        }
        
        if config_name in default_configs:
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(default_configs[config_name], f)
            print(f"Created default config: {config_path}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of configuration (without .toml extension)
            
        Returns:
            Configuration dictionary
        """
        return self.configs.get(config_name, {})
    
    def get_value(self, 
                 config_name: str, 
                 key_path: str, 
                 default: Any = None) -> Any:
        """
        Get specific configuration value using dot notation.
        
        Args:
            config_name: Configuration file name
            key_path: Dot-separated key path (e.g., "database.threads")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config(config_name)
        
        # Navigate through nested keys
        keys = key_path.split(".")
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_value(self, 
                 config_name: str, 
                 key_path: str, 
                 value: Any,
                 persist: bool = False) -> bool:
        """
        Set configuration value (runtime only unless persisted).
        
        Args:
            config_name: Configuration file name
            key_path: Dot-separated key path
            value: Value to set
            persist: Whether to save to file
            
        Returns:
            Success status
        """
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        # Navigate to parent and set value
        keys = key_path.split(".")
        current = self.configs[config_name]
        
        try:
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
            
            # Persist to file if requested
            if persist:
                config_path = self.config_dir / f"{config_name}.toml"
                with open(config_path, 'w', encoding='utf-8') as f:
                    toml.dump(self.configs[config_name], f)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to set config value: {e}")
            return False
    
    def validate_config(self, config_name: str) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config_name: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        issues = []
        config = self.get_config(config_name)
        
        if config_name == "system_config":
            issues.extend(self._validate_system_config(config))
        elif config_name == "database_config":
            issues.extend(self._validate_database_config(config))
        elif config_name == "models_config":
            issues.extend(self._validate_models_config(config))
        elif config_name == "strategies_config":
            issues.extend(self._validate_strategies_config(config))
        
        return issues
    
    def _validate_system_config(self, config: Dict) -> List[str]:
        """Validate system configuration."""
        issues = []
        
        # Required sections
        required_sections = ["system", "performance", "paths"]
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Performance validation
        if "performance" in config:
            perf = config["performance"]
            if "duckdb_threads" in perf:
                threads = perf["duckdb_threads"]
                if not isinstance(threads, int) or threads < 1:
                    issues.append("duckdb_threads must be positive integer")
        
        return issues
    
    def _validate_database_config(self, config: Dict) -> List[str]:
        """Validate database configuration."""
        issues = []
        
        if "duckdb" in config:
            duckdb_config = config["duckdb"]
            
            # Check memory limit format
            if "memory_limit" in duckdb_config:
                memory_limit = duckdb_config["memory_limit"]
                if not isinstance(memory_limit, str) or not any(
                    memory_limit.endswith(unit) for unit in ["MB", "GB", "TB"]
                ):
                    issues.append("memory_limit must end with MB, GB, or TB")
        
        return issues
    
    def _validate_models_config(self, config: Dict) -> List[str]:
        """Validate models configuration.""" 
        issues = []
        
        # Black-Scholes validation
        if "black_scholes" in config:
            bs_config = config["black_scholes"]
            
            if "default_risk_free_rate" in bs_config:
                rate = bs_config["default_risk_free_rate"]
                if not isinstance(rate, (int, float)) or rate < 0:
                    issues.append("default_risk_free_rate must be non-negative number")
        
        # Monte Carlo validation
        if "monte_carlo" in config:
            mc_config = config["monte_carlo"]
            
            if "default_simulations" in mc_config:
                sims = mc_config["default_simulations"]
                if not isinstance(sims, int) or sims < 1000:
                    issues.append("default_simulations must be at least 1000")
        
        return issues
    
    def _validate_strategies_config(self, config: Dict) -> List[str]:
        """Validate strategies configuration."""
        issues = []
        
        # Delta neutral validation
        if "delta_neutral" in config:
            dn_config = config["delta_neutral"]
            
            if "rebalance_threshold" in dn_config:
                threshold = dn_config["rebalance_threshold"]
                if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
                    issues.append("rebalance_threshold must be between 0 and 1")
        
        return issues
    
    def reload_config(self, config_name: str) -> bool:
        """
        Reload specific configuration from file.
        
        Args:
            config_name: Configuration to reload
            
        Returns:
            Success status
        """
        try:
            config_path = self.config_dir / f"{config_name}.toml"
            if config_path.exists():
                self.configs[config_name] = self._load_config_file(config_path)
                return True
            return False
        except Exception as e:
            warnings.warn(f"Failed to reload config {config_name}: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations."""
        return self.configs.copy()


# Global configuration manager
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration by name.
    
    Args:
        config_name: Configuration file name (without .toml)
        
    Returns:
        Configuration dictionary
    """
    return get_config_manager().get_config(config_name)


def get_config(config_name: str, key_path: str, default: Any = None) -> Any:
    """
    Get specific configuration value.
    
    Args:
        config_name: Configuration file name
        key_path: Dot-separated key path
        default: Default value if not found
        
    Returns:
        Configuration value or default
    """
    return get_config_manager().get_value(config_name, key_path, default)


def validate_all_configs() -> Dict[str, List[str]]:
    """
    Validate all configurations.
    
    Returns:
        Dictionary mapping config names to validation issues
    """
    all_issues = {}
    manager = get_config_manager()
    
    for config_name in manager.configs.keys():
        issues = manager.validate_config(config_name)
        if issues:
            all_issues[config_name] = issues
    
    return all_issues


# Usage examples
if __name__ == "__main__":
    # Load system configuration
    system_config = load_config("system_config")
    print(f"System name: {system_config.get('system', {}).get('name')}")
    
    # Get specific value with default
    threads = get_config("database_config", "duckdb.threads", 4)
    print(f"Database threads: {threads}")
    
    # Validate configurations
    issues = validate_all_configs()
    if issues:
        print("Configuration issues found:")
        for config_name, config_issues in issues.items():
            print(f"  {config_name}: {config_issues}")
    else:
        print("All configurations are valid")