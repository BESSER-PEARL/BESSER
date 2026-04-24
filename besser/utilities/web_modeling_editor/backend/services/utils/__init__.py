"""
Utils module for utility functions.
"""

from .layout_calculator import (
    calculate_center_point,
    determine_connection_direction,
    calculate_connection_points,
    calculate_path_points,
    calculate_relationship_bounds,
)
from .resource_manager import (
    cleanup_temp_resources,
    validate_generator,
)
from .agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    load_default_agent_recommendation_config,
    extract_json_object,
    normalize_recommended_agent_config,
)
from .agent_config_manual_mapping_utils import (
    MANUAL_AGENT_CONFIG_MAPPING,
    get_manual_agent_config_mapping,
    build_manual_mapping_recommendation,
)

# NOTE: ``user_profile_utils`` is intentionally NOT re-exported here.
# It imports ``backend.config`` (for ``get_generator_info``), which in turn
# imports ``BAFGenerator`` — and BAFGenerator imports ``services.converters``.
# Because ``services/converters/__init__.py`` itself imports from
# ``services.utils`` (for the layout helpers), eagerly loading
# ``user_profile_utils`` at package import time closes that loop and raises
# ``ImportError: cannot import name 'agent_buml_to_json' from partially
# initialized module 'services.converters'``.
#
# Consumers that need the user-profile helpers import them directly from the
# submodule:
#
#     from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils \
#         import generate_user_profile_document

__all__ = [
    "calculate_center_point",
    "determine_connection_direction",
    "calculate_connection_points",
    "calculate_path_points",
    "calculate_relationship_bounds",
    "cleanup_temp_resources",
    "validate_generator",
    "RECOMMENDATION_ALLOWED_VALUES",
    "load_default_agent_recommendation_config",
    "extract_json_object",
    "normalize_recommended_agent_config",
    "MANUAL_AGENT_CONFIG_MAPPING",
    "get_manual_agent_config_mapping",
    "build_manual_mapping_recommendation",
]
