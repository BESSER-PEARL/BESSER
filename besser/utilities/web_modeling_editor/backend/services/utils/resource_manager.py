"""
Resource management utilities for the BESSER backend.
"""


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
