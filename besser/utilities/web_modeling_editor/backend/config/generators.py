"""
Generator configuration and metadata for the BESSER backend.
"""

from typing import Dict, Any, NamedTuple, Optional

from besser.generators.registry import GENERATORS


class GeneratorInfo(NamedTuple):
    """Information about a generator."""
    generator_class: Any
    output_type: str  # "file" or "zip"
    file_extension: str
    category: str
    requires_class_diagram: bool = True
    # Non-class diagram type this generator consumes (e.g. ``NNDiagram``,
    # ``QuantumCircuitDiagram``, ``AgentDiagram``). ``None`` means the
    # generator works off the project's active class diagram. Used by the
    # project endpoint to dispatch to the right diagram without branching
    # on generator-name string literals.
    required_diagram_type: Optional[str] = None
    # Name of the downloaded file (``Content-Disposition``). ``None`` falls
    # back to ``<generator_type>_output<file_extension>``; see
    # ``get_filename_for_generator``.
    download_filename: Optional[str] = None
    # Single-file generators only: ship the file under the name the generator
    # chose instead of ``download_filename`` (e.g. Supabase's migration-style
    # ``<timestamp>_<slug>.sql``, whose name is meaningful).
    keep_generated_filename: bool = False
    # What the generator needs besides one diagram, so ``/generate-output``
    # refuses it (``GeneratorSpec.needs_beyond_one_diagram``).
    needs_beyond_one_diagram: Optional[str] = None


def _supported_generators() -> Dict[str, GeneratorInfo]:
    """The web editor's generators, from ``besser.generators.registry``."""
    supported: Dict[str, GeneratorInfo] = {}
    for spec in GENERATORS:
        supported[spec.id] = GeneratorInfo(
            generator_class=spec.load_class(),
            output_type=spec.output_type,
            file_extension=spec.file_extension,
            category=spec.category,
            requires_class_diagram=spec.requires_class_diagram,
            required_diagram_type=spec.required_diagram_type,
            download_filename=spec.download_filename,
            keep_generated_filename=spec.keep_generated_filename,
            needs_beyond_one_diagram=spec.needs_beyond_one_diagram,
        )
    return supported


SUPPORTED_GENERATORS: Dict[str, GeneratorInfo] = _supported_generators()


# Filename prefix for NN generator output. The NN generator strips the
# trailing ".py" from its configured file_name and appends
# ``_{generation_type}.py``, so the artifact for ``pytorch`` /
# ``tensorflow`` is e.g. ``pytorch_nn_subclassing.py`` /
# ``tf_nn_sequential.py``. Centralizing the mapping here keeps the router
# free of hard-coded string switches.
_NN_FILENAME_PREFIX: Dict[str, str] = {
    "pytorch": "pytorch",
    "tensorflow": "tf",
}


def get_nn_filename(generator_type: str, generation_type: str) -> str:
    """Return the on-disk filename produced by an NN generator for the given
    ``generation_type`` variant (``"subclassing"`` or ``"sequential"``)."""
    prefix = _NN_FILENAME_PREFIX[generator_type]
    return f"{prefix}_nn_{generation_type}.py"


def get_generator_info(generator_type: str) -> GeneratorInfo | None:
    """Get generator information by type."""
    return SUPPORTED_GENERATORS.get(generator_type)


def get_filename_for_generator(generator_type: str, base_name: str = "output") -> str:
    """Return the download filename for a generator.

    Uses ``GeneratorInfo.download_filename`` when set, otherwise
    ``<generator_type>_output<file_extension>``; an unknown generator gets
    ``<base_name>.txt``.
    """
    info = get_generator_info(generator_type)
    if not info:
        return f"{base_name}.txt"
    return info.download_filename or f"{generator_type}_output{info.file_extension}"


def is_generator_supported(generator_type: str) -> bool:
    """Check if a generator type is supported."""
    return generator_type in SUPPORTED_GENERATORS
