"""
Configuration Management System for Financial Complaint Analysis RAG System

This module provides centralized configuration management with environment-specific
settings, validation, and dynamic updates.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_model: str = "all-MiniLM-L6-v2"
    generation_model: str = "microsoft/DialoGPT-medium"
    use_mock_generator: bool = True
    max_tokens: int = 500
    temperature: float = 0.7
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Vector store configuration settings."""
    chunk_size: int = 300
    chunk_overlap: int = 50
    similarity_threshold: float = 0.3
    index_type: str = "faiss"
    distance_metric: str = "cosine"


@dataclass
class RAGConfig:
    """RAG pipeline configuration settings."""
    top_k_retrieval: int = 5
    max_context_length: int = 2000
    confidence_threshold: float = 0.1
    enable_reranking: bool = False
    response_timeout: int = 30


@dataclass
class UIConfig:
    """User interface configuration settings."""
    app_title: str = "Financial Complaint Analysis RAG System"
    app_icon: str = "🏦"
    default_port: int = 8501
    theme: str = "light"
    show_source_metadata: bool = True
    max_chat_history: int = 50


@dataclass
class DataConfig:
    """Data processing configuration settings."""
    target_products: List[str] = None
    min_text_length: int = 50
    max_text_length: int = 5000
    filter_empty_narratives: bool = True
    sample_size: Optional[int] = None
    
    def __post_init__(self):
        if self.target_products is None:
            self.target_products = [
                "Credit card",
                "Personal loan", 
                "Buy now pay later",
                "Savings account",
                "Money transfers"
            ]


@dataclass
class SystemConfig:
    """Complete system configuration."""
    model: ModelConfig
    vector_store: VectorStoreConfig
    rag: RAGConfig
    ui: UIConfig
    data: DataConfig
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config = None
        self._config_file = None
    
    def load_config(self, config_file: Optional[str] = None, environment: str = "development") -> SystemConfig:
        """Load configuration from file or create default."""
        if config_file is None:
            config_file = self.config_dir / f"{environment}.yaml"
        
        self._config_file = Path(config_file)
        
        if self._config_file.exists():
            self._config = self._load_from_file(self._config_file)
            logger.info(f"Configuration loaded from: {self._config_file}")
        else:
            self._config = self._create_default_config(environment)
            self.save_config()
            logger.info(f"Default configuration created: {self._config_file}")
        
        return self._config
    
    def _load_from_file(self, config_file: Path) -> SystemConfig:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return SystemConfig(
                model=ModelConfig(**data.get('model', {})),
                vector_store=VectorStoreConfig(**data.get('vector_store', {})),
                rag=RAGConfig(**data.get('rag', {})),
                ui=UIConfig(**data.get('ui', {})),
                data=DataConfig(**data.get('data', {})),
                environment=data.get('environment', 'development'),
                debug=data.get('debug', False),
                log_level=data.get('log_level', 'INFO')
            )
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
            return self._create_default_config()
    
    def _create_default_config(self, environment: str = "development") -> SystemConfig:
        """Create default configuration."""
        return SystemConfig(
            model=ModelConfig(),
            vector_store=VectorStoreConfig(),
            rag=RAGConfig(),
            ui=UIConfig(),
            data=DataConfig(),
            environment=environment,
            debug=environment == "development",
            log_level="DEBUG" if environment == "development" else "INFO"
        )
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file."""
        if config_file is None:
            config_file = self._config_file or self.config_dir / "development.yaml"
        
        config_file = Path(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(asdict(self._config), f, default_flow_style=False, indent=2)
                else:
                    json.dump(asdict(self._config), f, indent=2)
            
            logger.info(f"Configuration saved to: {config_file}")
        except Exception as e:
            logger.error(f"Error saving config to {config_file}: {e}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if self._config is None:
            self._config = self.load_config()
        
        # Update nested configuration
        for section, values in updates.items():
            if hasattr(self._config, section):
                section_config = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section}.{key}")
            else:
                logger.warning(f"Unknown config section: {section}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        config = self.get_config()
        
        # Validate model config
        if config.model.chunk_size <= 0:
            issues.append("Model chunk_size must be positive")
        
        if not (0.0 <= config.model.temperature <= 2.0):
            issues.append("Model temperature must be between 0.0 and 2.0")
        
        # Validate vector store config
        if config.vector_store.chunk_size <= config.vector_store.chunk_overlap:
            issues.append("Chunk size must be larger than chunk overlap")
        
        if not (0.0 <= config.vector_store.similarity_threshold <= 1.0):
            issues.append("Similarity threshold must be between 0.0 and 1.0")
        
        # Validate RAG config
        if config.rag.top_k_retrieval <= 0:
            issues.append("Top-k retrieval must be positive")
        
        if config.rag.max_context_length <= 0:
            issues.append("Max context length must be positive")
        
        # Validate UI config
        if not (1024 <= config.ui.default_port <= 65535):
            issues.append("UI port must be between 1024 and 65535")
        
        # Validate data config
        if config.data.min_text_length >= config.data.max_text_length:
            issues.append("Min text length must be less than max text length")
        
        return issues
    
    def get_environment_configs(self) -> Dict[str, str]:
        """Get available environment configurations."""
        configs = {}
        for config_file in self.config_dir.glob("*.yaml"):
            env_name = config_file.stem
            configs[env_name] = str(config_file)
        return configs
    
    def create_environment_config(self, environment: str, base_environment: str = "development"):
        """Create a new environment configuration based on existing one."""
        base_config_file = self.config_dir / f"{base_environment}.yaml"
        new_config_file = self.config_dir / f"{environment}.yaml"
        
        if base_config_file.exists():
            # Load base config and modify environment
            base_config = self._load_from_file(base_config_file)
            base_config.environment = environment
            
            # Environment-specific adjustments
            if environment == "production":
                base_config.debug = False
                base_config.log_level = "WARNING"
                base_config.model.use_mock_generator = False
            elif environment == "testing":
                base_config.debug = True
                base_config.log_level = "DEBUG"
                base_config.model.use_mock_generator = True
                base_config.data.sample_size = 1000
            
            # Save new config
            self._config = base_config
            self.save_config(new_config_file)
            logger.info(f"Created {environment} config based on {base_environment}")
        else:
            logger.error(f"Base configuration not found: {base_config_file}")
    
    def export_config(self, format: str = "yaml") -> str:
        """Export current configuration as string."""
        config = self.get_config()
        
        if format.lower() == "json":
            return json.dumps(asdict(config), indent=2)
        else:
            return yaml.dump(asdict(config), default_flow_style=False, indent=2)
    
    def import_config(self, config_str: str, format: str = "yaml"):
        """Import configuration from string."""
        try:
            if format.lower() == "json":
                data = json.loads(config_str)
            else:
                data = yaml.safe_load(config_str)
            
            self._config = SystemConfig(
                model=ModelConfig(**data.get('model', {})),
                vector_store=VectorStoreConfig(**data.get('vector_store', {})),
                rag=RAGConfig(**data.get('rag', {})),
                ui=UIConfig(**data.get('ui', {})),
                data=DataConfig(**data.get('data', {})),
                environment=data.get('environment', 'development'),
                debug=data.get('debug', False),
                log_level=data.get('log_level', 'INFO')
            )
            
            logger.info("Configuration imported successfully")
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(environment: str = None) -> SystemConfig:
    """Get system configuration."""
    manager = get_config_manager()
    if environment:
        return manager.load_config(environment=environment)
    return manager.get_config()


def main():
    """Command-line interface for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument('--environment', '-e', default='development', help='Environment name')
    parser.add_argument('--create', help='Create new environment config')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--export', choices=['yaml', 'json'], help='Export configuration')
    parser.add_argument('--list', action='store_true', help='List available configurations')
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.list:
        configs = manager.get_environment_configs()
        print("Available configurations:")
        for env, path in configs.items():
            print(f"  {env}: {path}")
        return
    
    if args.create:
        manager.create_environment_config(args.create, args.environment)
        return
    
    # Load configuration
    config = manager.load_config(environment=args.environment)
    
    if args.validate:
        issues = manager.validate_config()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
    
    if args.export:
        print(manager.export_config(args.export))


if __name__ == "__main__":
    main()