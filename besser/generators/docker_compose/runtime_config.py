"""Resolve saved BAF listener settings for Docker build contexts."""

from typing import Optional

import yaml

from besser.generators.agents.agent_personalization import flatten_agent_config_structure


def _mapping(parent: dict, key: str) -> dict:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"BAF config section '{key}' must be a mapping")
    return value


def _port(section: dict, default: int, label: str) -> int:
    value = section.get('port', default)
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"BAF {label} port must be an integer between 1 and 65535")
    section['port'] = value
    return value


def resolve_runtime_yaml(source: Optional[str]) -> tuple:
    """Return container YAML plus the three effective BAF listener ports.

    No saved YAML retains the existing BAF default-template path. Saved
    YAML is parsed into a new object; all non-listener values are retained.
    """
    if source is None or (isinstance(source, str) and not source.strip()):
        return None, {'a2a': 8000, 'websocket': 8765, 'streamlit': 5000}
    if not isinstance(source, str):
        raise ValueError('BAF configYaml must be a non-empty YAML mapping when supplied')
    try:
        config = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError('BAF configYaml contains invalid YAML') from exc
    if not isinstance(config, dict):
        raise ValueError('BAF configYaml must contain a YAML mapping')
    platforms = _mapping(config, 'platforms')
    websocket = _mapping(platforms, 'websocket')
    streamlit = _mapping(websocket, 'streamlit')
    a2a = _mapping(platforms, 'a2a')
    ports = {
        'a2a': _port(a2a, 8000, 'A2A'),
        'websocket': _port(websocket, 8765, 'WebSocket'),
        'streamlit': _port(streamlit, 5000, 'Streamlit'),
    }
    websocket['host'] = '0.0.0.0'
    streamlit['host'] = '0.0.0.0'
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True), ports


def human_listener_ports(agent, config, ports: dict) -> list:
    """Mirror the existing BAF template's platform selection for publishing."""
    flat = flatten_agent_config_structure(config) if isinstance(config, dict) else {}
    selected = flat.get('agentPlatform')
    if selected == 'telegram':
        return []
    if selected == 'websocket':
        return [(8765, ports['websocket'])]
    if selected != 'streamlit':
        for platform in agent.platforms:
            kind = type(platform).__name__
            if kind == 'TelegramPlatform':
                return []
            if kind in ('WebSocketPlatform', 'StreamlitPlatform'):
                break
        else:
            # The no-platform fallback in the BAF template is WebSocket + UI.
            pass
    return [(5001, ports['streamlit']), (8765, ports['websocket'])]
