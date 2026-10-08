"""Materialize isolated declared components for Docker BAF generation."""

import ast
from pathlib import Path


def _entrypoint(tool) -> str:
    try:
        tree = ast.parse(tool.code or '')
    except SyntaxError as exc:
        raise ValueError(f"Tool '{tool.name}' contains invalid Python") from exc
    functions = [node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and not node.name.startswith('_')]
    if len(functions) != len(set(functions)):
        raise ValueError(f"Tool '{tool.name}' repeats a public function definition")
    if tool.name in functions:
        return tool.name
    if len(functions) == 1:
        return functions[0]
    raise ValueError(
        f"Tool '{tool.name}' needs one public top-level function or a function matching its declared name"
    )


def materialize_deployment_components(agent, output_dir: str) -> dict:
    """Write authored code/content unchanged and return registration bindings."""
    root = Path(output_dir)
    bindings = {'tools': [], 'skills': []}
    # Validate all tool entrypoints before writing any component assets.
    tools = [(tool, _entrypoint(tool)) for tool in (agent.tools or [])]
    for index, (tool, function) in enumerate(tools):
        relative = f'declared_tools/tool_{index:03d}.py'
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tool.code, encoding='utf-8')
        bindings['tools'].append({
            'name': tool.name, 'description': tool.description,
            'path': relative, 'function': function,
        })
    for index, skill in enumerate(agent.skills or []):
        relative = f'skills/skill_{index:03d}.md'
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(skill.content, encoding='utf-8')
        bindings['skills'].append({
            'name': skill.name, 'description': skill.description, 'path': relative,
        })
    return bindings