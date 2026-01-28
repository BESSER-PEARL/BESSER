"""
GUI Diagram Processors Package

This package provides modular processors for converting GrapesJS JSON to BUML GUI models.
Each module handles a specific aspect of the conversion process.
"""

from .processor import process_gui_diagram

__all__ = ['process_gui_diagram']
