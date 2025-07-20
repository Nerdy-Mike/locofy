"""
Service layer initialization.
This file ensures proper module initialization and dependency management.
"""

from .llm_service import LLMUIDetectionService

__all__ = [
    "LLMUIDetectionService",
]
