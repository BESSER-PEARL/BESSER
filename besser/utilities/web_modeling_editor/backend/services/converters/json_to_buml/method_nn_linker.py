"""
Link class-diagram methods to the NN diagrams that implement them.

The editor stores a method's network as ``neuralNetworkId`` (the NNDiagram's
id); ``process_class_diagram`` keeps it in ``domain_model.method_diagram_refs``.
This module resolves those ids against a project's NNDiagrams and sets
``Method.neural_network`` so generators can use the actual ``NN`` model.
"""

import logging

from .nn_diagram_processor import process_nn_diagram

logger = logging.getLogger(__name__)


def link_method_neural_networks(domain_model, nn_diagrams, processed_nn_models=None):
    """Set ``method.neural_network`` for every method that references an NNDiagram.

    Args:
        domain_model: DomainModel produced by ``process_class_diagram``.
        nn_diagrams: the project's NNDiagram inputs (objects with ``id``, ``title``
            and ``model_dump()``).
        processed_nn_models: optional ``{diagram id: NN}`` of networks already
            processed, reused so a network is converted only once.

    Returns:
        int: the number of methods linked.
    """
    refs = getattr(domain_model, "method_diagram_refs", None) or {}
    wanted = {key: ref.get("neuralNetworkId") for key, ref in refs.items() if ref.get("neuralNetworkId")}
    if not wanted:
        return 0

    by_id = {d.id: d for d in nn_diagrams if getattr(d, "id", None)}
    # Imported BUML files carry the NN's name instead of a diagram id.
    by_title = {d.title: d for d in nn_diagrams if getattr(d, "title", None)}
    cache = processed_nn_models if processed_nn_models is not None else {}

    linked = 0
    for (class_name, method_name), nn_ref in wanted.items():
        cls = domain_model.get_class_by_name(class_name)
        method = next((m for m in cls.methods if m.name == method_name), None) if cls else None
        diagram = by_id.get(nn_ref) or by_title.get(nn_ref)
        if method is None or diagram is None:
            logger.warning(
                "Method '%s.%s' references neural network '%s', which is not in the project.",
                class_name, method_name, nn_ref,
            )
            continue
        key = diagram.id or id(diagram)
        if key not in cache:
            cache[key] = process_nn_diagram(diagram.model_dump())
        method.neural_network = cache[key]
        linked += 1
    return linked
