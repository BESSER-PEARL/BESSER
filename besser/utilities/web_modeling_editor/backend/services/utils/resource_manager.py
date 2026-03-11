"""
Resource management utilities for the BESSER backend.
"""

import os
import sys
import shutil
from typing import Optional


def cleanup_temp_resources(temp_dir: Optional[str] = None):
    """
    Clean up temporary resources.

    Args:
        temp_dir: Temporary directory to clean up. Only this directory
                  (and its contents) will be removed.
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Remove temp directory from sys.path if it exists
    if temp_dir and temp_dir in sys.path:
        sys.path.remove(temp_dir)


def validate_generator(generator_type: str, supported_generators: dict) -> bool:
    """
    Validate if the generator type is supported.
    
    Args:
        generator_type: The type of generator to validate
        supported_generators: Dictionary of supported generators
        
    Returns:
        bool: True if generator is supported, False otherwise
    """
    return generator_type in supported_generators
