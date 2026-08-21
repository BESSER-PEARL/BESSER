"""
Method parsing utilities for converting JSON to BUML format.
"""

import logging

from besser.BUML.metamodel.structural import Enumeration, Class
from besser.utilities.web_modeling_editor.backend.constants.constants import VISIBILITY_MAP, VALID_PRIMITIVE_TYPES

logger = logging.getLogger(__name__)


def parse_method(method_str, domain_model=None, type_lookup=None):
    """
    Parse a method string to extract visibility, name, parameters, and return type.

    Args:
        method_str: The raw method signature string.
        domain_model: Optional DomainModel for resolving user-defined types.
        type_lookup: Optional pre-built ``{name: type_obj}`` dict for O(1)
            type resolution.  When provided, *domain_model* is still accepted
            but the lookup dict is preferred.

    Examples:
    "+ notify(sms: str = 'message')" -> ("public", "notify", [{"name": "sms", "type": "str", "default": "message"}], None)
    "- findBook(title: str): Book" -> ("private", "findBook", [{"name": "title", "type": "str"}], "Book")
    "validate()" -> ("public", "validate", [], None)
    """

    # Default values
    visibility = "public"
    parameters = []
    return_type = None

    # Check if this is actually a method (contains parentheses)
    if '(' not in method_str:
        return visibility, method_str, parameters, return_type

    # Extract visibility if present
    method_str = method_str.strip()
    if method_str.startswith(tuple(VISIBILITY_MAP.keys())):
        visibility = VISIBILITY_MAP.get(method_str[0], "public")
        method_str = method_str[1:].lstrip()

    # Parse method by finding the first '(' and the last ')' to handle
    # parenthesized default values like method(x: str = "a(b)")
    first_paren = method_str.index('(')
    last_paren = method_str.rindex(')')
    method_name = method_str[:first_paren].strip()
    params_str = method_str[first_paren + 1:last_paren].strip()
    rest = method_str[last_paren + 1:].strip()
    return_type = None
    if rest.startswith(':'):
        return_type = rest[1:].strip() or None

    # Parse parameters if present
    if params_str:
        # Handle nested parentheses in default values
        param_list = []
        current_param = []
        paren_count = 0

        for char in params_str + ',':
            if char == '(' and paren_count >= 0:
                paren_count += 1
                current_param.append(char)
            elif char == ')' and paren_count > 0:
                paren_count -= 1
                current_param.append(char)
            elif char == ',' and paren_count == 0:
                param_list.append(''.join(current_param).strip())
                current_param = []
            else:
                current_param.append(char)

        for param in param_list:
            if not param:
                continue

            param_dict = {'name': param, 'type': 'any'}
            if ':' not in param and '=' not in param:
                logger.warning("Parameter '%s' in method '%s' has no type annotation, defaulting to 'any'.", param.strip(), method_name)

            # Handle parameter with default value
            if '=' in param:
                param_parts = param.split('=', 1)
                param_name_type = param_parts[0].strip()
                default_value = param_parts[1].strip().strip('"\'')

                if ':' in param_name_type:
                    param_name, param_type = [p.strip() for p in param_name_type.split(':')]
                    param_dict.update({
                        'name': param_name,
                        'type': VALID_PRIMITIVE_TYPES.get(param_type.lower(), param_type),
                        'default': default_value
                    })
                else:
                    param_dict.update({
                        'name': param_name_type,
                        'default': default_value
                    })

            # Handle parameter with type annotation
            elif ':' in param:
                param_name, param_type = [p.strip() for p in param.split(':')]

                # Handle the type — use O(1) lookup when available
                is_user_type = False
                if type_lookup is not None:
                    is_user_type = param_type in type_lookup
                elif domain_model:
                    is_user_type = any(isinstance(t, (Enumeration, Class)) and t.name == param_type for t in domain_model.types)

                if is_user_type:
                    type_param = param_type
                else:
                    type_param = VALID_PRIMITIVE_TYPES.get(param_type.lower(), None)
                    if type_param is None:
                        raise ValueError(f"Invalid type '{param_type}' for the parameter '{param_name}'")

                param_dict.update({
                    'name': param_name,
                    'type': type_param
                })
            else:
                param_dict['name'] = param.strip()

            parameters.append(param_dict)

    # Clean up return type if present
    type_return = None
    if return_type:
        return_type = return_type.strip()
        # Keep the original return type if it's not a primitive type — use O(1) lookup when available
        is_user_return = False
        if type_lookup is not None:
            is_user_return = return_type in type_lookup
        elif domain_model:
            is_user_return = any(isinstance(t, (Enumeration, Class)) and t.name == return_type for t in domain_model.types)

        if is_user_return:
            type_return = return_type
        else:
            type_return = VALID_PRIMITIVE_TYPES.get(return_type.lower(), None)
            if type_return is None:
                raise ValueError(f"Invalid return type '{return_type}' for the method '{method_name}'")

    return visibility, method_name, parameters, type_return
