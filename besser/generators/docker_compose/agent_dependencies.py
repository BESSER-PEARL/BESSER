"""Select BAF dependency groups from the agent code rendered for deployment."""

import ast


def baf_dependency_extras(agent_code: str) -> tuple[str, ...]:
    """Keep API/RAG dependencies and add only the emitted local ML backends.

    Inspect constructor calls after BAF generation so configuration precedence,
    state classifiers and personalized graphs follow the existing BAF template.
    Imports alone do not require a backend: the template imports every provider.
    Dynamic classifier arguments conservatively retain both supported backends.
    Arbitrary third-party dependencies in authored code remain user supplied.
    """
    extras = {'extras', 'llms'}
    for node in ast.walk(ast.parse(agent_code)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name == 'LLMHuggingFace':
            # BAF's local text-generation pipeline uses PyTorch. The API and
            # Ollama wrappers run inference outside the generated container.
            extras.add('torch')
        elif name == 'SimpleIntentClassifierConfiguration':
            framework = next((kw.value for kw in node.keywords if kw.arg == 'framework'),
                             node.args[0] if node.args else ast.Constant('pytorch'))
            if any(kw.arg is None for kw in node.keywords):
                extras.update(('torch', 'tensorflow'))
            elif isinstance(framework, ast.Constant) and framework.value == 'pytorch':
                extras.add('torch')
            elif isinstance(framework, ast.Constant) and framework.value == 'tensorflow':
                extras.add('tensorflow')
            else:
                extras.update(('torch', 'tensorflow'))
    return tuple(extra for extra in ('extras', 'llms', 'torch', 'tensorflow') if extra in extras)
