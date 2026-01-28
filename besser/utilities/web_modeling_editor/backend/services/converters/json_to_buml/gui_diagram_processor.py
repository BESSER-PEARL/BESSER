"""
GUI Diagram Processor - Legacy compatibility module.

This module maintains backward compatibility by importing from the new modular structure.
All functionality has been reorganized into gui_processors package for better maintainability.

New structure:
- gui_processors/
   __init__.py           - Package initialization
   constants.py          - Constants and mappings (TEXT_TAGS, CONTAINER_TAGS, etc.)
   utils.py              - Utility functions (sanitize_name, extract_text_content, etc.)
   styling.py            - CSS/Style processing (styling_from_css, resolve_component_styling)
   component_helpers.py  - Component detection helpers (has_menu_structure, extract_menu_items)
   component_parsers.py  - Basic component parsers (parse_button, parse_form, parse_menu, etc.)
   chart_parsers.py      - Chart component parsers (parse_line_chart, parse_bar_chart, etc.)
   processor.py          - Main orchestration logic (process_gui_diagram)

Usage:
    from gui_diagram_processor import process_gui_diagram
    gui_model = process_gui_diagram(gui_diagram, class_model, domain_model)
"""

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    process_gui_diagram,
)

__all__ = ['process_gui_diagram']
