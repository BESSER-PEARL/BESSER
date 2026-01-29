"""
Configuration module for the BESSER backend.
"""

from .generators import (
    SUPPORTED_GENERATORS,
    GeneratorInfo,
    get_generator_info,
    get_filename_for_generator,
    is_generator_supported,
)

__all__ = [
    "SUPPORTED_GENERATORS",
    "GeneratorInfo",
    "get_generator_info",
    "get_filename_for_generator",
    "is_generator_supported",
]
