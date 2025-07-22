import logging
import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from models.validation_models import ValidationConfig

logger = logging.getLogger(__name__)


class AppConfig(BaseSettings):
    """Application configuration with environment variable support"""

    # OpenAI API configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # LLM Validation settings
    llm_validation_enabled: bool = Field(True, env="LLM_VALIDATION_ENABLED")
    llm_validation_timeout: int = Field(10, env="LLM_VALIDATION_TIMEOUT", ge=1, le=60)
    llm_validation_confidence_threshold: float = Field(
        0.7, env="LLM_VALIDATION_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0
    )
    llm_validation_max_image_size_mb: float = Field(
        5.0, env="LLM_VALIDATION_MAX_IMAGE_SIZE_MB", gt=0
    )
    llm_validation_fallback_on_error: bool = Field(
        True, env="LLM_VALIDATION_FALLBACK_ON_ERROR"
    )
    llm_validation_cache_results: bool = Field(True, env="LLM_VALIDATION_CACHE_RESULTS")

    # MCP (Model Context Protocol) settings
    mcp_enabled: bool = Field(True, env="MCP_ENABLED")
    mcp_timeout: int = Field(30, env="MCP_TIMEOUT", ge=10, le=120)
    mcp_fallback_to_direct_api: bool = Field(True, env="MCP_FALLBACK_TO_DIRECT_API")
    mcp_server_command: str = Field("npx", env="MCP_SERVER_COMMAND")
    mcp_server_args: str = Field(
        "-y,@anthropic-ai/mcp-server-openai", env="MCP_SERVER_ARGS"
    )
    mcp_health_check_interval: int = Field(
        60, env="MCP_HEALTH_CHECK_INTERVAL", ge=30, le=300
    )
    mcp_max_context_size: int = Field(10, env="MCP_MAX_CONTEXT_SIZE", ge=1, le=50)

    # File storage settings
    data_directory: str = Field("data", env="DATA_DIRECTORY")
    temp_directory: Optional[str] = Field(None, env="TEMP_DIRECTORY")
    temp_file_cleanup_interval: int = Field(
        3600, env="TEMP_FILE_CLEANUP_INTERVAL", gt=0
    )

    # Upload limits
    max_file_size_mb: float = Field(10.0, env="MAX_FILE_SIZE_MB", gt=0)
    allowed_file_types: str = Field(
        "image/jpeg,image/jpg,image/png,image/gif,image/bmp", env="ALLOWED_FILE_TYPES"
    )

    # API settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT", ge=1, le=65535)
    api_debug: bool = Field(False, env="API_DEBUG")

    # Logging settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    def get_validation_config(self) -> ValidationConfig:
        """Create ValidationConfig from application settings"""
        return ValidationConfig(
            enabled=self.llm_validation_enabled,
            confidence_threshold=self.llm_validation_confidence_threshold,
            timeout_seconds=self.llm_validation_timeout,
            max_image_size_mb=self.llm_validation_max_image_size_mb,
            fallback_on_error=self.llm_validation_fallback_on_error,
            cache_validation_results=self.llm_validation_cache_results,
            temp_file_cleanup_interval=self.temp_file_cleanup_interval,
        )

    def get_allowed_file_types_list(self) -> list:
        """Get allowed file types as a list"""
        return [t.strip() for t in self.allowed_file_types.split(",")]

    def setup_logging(self):
        """Configure application logging"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(self.log_file)] if self.log_file else []),
            ],
        )

        logger.info(f"Logging configured: level={self.log_level}")


class ConfigManager:
    """Singleton configuration manager"""

    _instance: Optional[AppConfig] = None

    @classmethod
    def get_config(cls) -> AppConfig:
        """Get or create application configuration"""
        if cls._instance is None:
            try:
                cls._instance = AppConfig()
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

        return cls._instance

    @classmethod
    def reload_config(cls) -> AppConfig:
        """Reload configuration from environment"""
        cls._instance = None
        return cls.get_config()


# Convenience function for getting config
def get_config() -> AppConfig:
    """Get application configuration"""
    return ConfigManager.get_config()


# Environment validation functions
def validate_environment():
    """Validate that all required environment variables are set"""
    config = get_config()

    errors = []

    # Check OpenAI API key
    if not config.openai_api_key:
        errors.append("OPENAI_API_KEY is required")

    # Check data directory exists or can be created
    try:
        os.makedirs(config.data_directory, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create data directory '{config.data_directory}': {e}")

    # Check temp directory if specified
    if config.temp_directory:
        try:
            os.makedirs(config.temp_directory, exist_ok=True)
        except Exception as e:
            errors.append(
                f"Cannot create temp directory '{config.temp_directory}': {e}"
            )

    if errors:
        error_msg = "Environment validation failed:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Environment validation passed")


def create_sample_env_file(filepath: str = ".env.sample"):
    """Create a sample environment file with all configuration options"""

    sample_content = """# Locofy UI Component Labeling System Configuration

# OpenAI API Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# LLM Validation Settings
LLM_VALIDATION_ENABLED=true
LLM_VALIDATION_TIMEOUT=10
LLM_VALIDATION_CONFIDENCE_THRESHOLD=0.7
LLM_VALIDATION_MAX_IMAGE_SIZE_MB=5.0
LLM_VALIDATION_FALLBACK_ON_ERROR=true
LLM_VALIDATION_CACHE_RESULTS=true

# MCP (Model Context Protocol) Settings
MCP_ENABLED=true
MCP_TIMEOUT=30
MCP_FALLBACK_TO_DIRECT_API=true
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y,@anthropic-ai/mcp-server-openai
MCP_HEALTH_CHECK_INTERVAL=60
MCP_MAX_CONTEXT_SIZE=10

# File Storage Settings
DATA_DIRECTORY=data
TEMP_DIRECTORY=
TEMP_FILE_CLEANUP_INTERVAL=3600

# Upload Limits
MAX_FILE_SIZE_MB=10.0
ALLOWED_FILE_TYPES=image/jpeg,image/jpg,image/png,image/gif,image/bmp

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=
"""

    try:
        with open(filepath, "w") as f:
            f.write(sample_content)
        logger.info(f"Sample environment file created: {filepath}")
    except Exception as e:
        logger.error(f"Failed to create sample env file: {e}")
        raise
