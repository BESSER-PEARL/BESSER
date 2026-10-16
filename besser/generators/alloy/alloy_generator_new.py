import tempfile

from besser.BUML.metamodel.structural.structural import AssociationClass, DomainModel
from besser.generators import GeneratorInterface
from besser.generators.alloy.alloy_translator_new import BumlToAlloy


class AlloyGeneratorNew(GeneratorInterface):
    """Writes the Alloy specification of a B-UML domain model and its OCL invariants.

    ``model.als`` holds the specification built by :class:`BumlToAlloy`; the library modules
    it opens are written next to it.

    Raises:
        NotImplementedError: when the model has association classes.
        OclTranslationError: when an OCL invariant cannot be translated.
    """

    def __init__(self, model: DomainModel, output_dir: str | None = None):
        if output_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            output_dir = self._temp_dir.name
        super().__init__(model, output_dir)
        self.file_path = None

    def generate(self, scope: int = 5) -> None:
        """Writes the Alloy specification of the model to ``<output_dir>/model.als``.

        Args:
            scope: the scope of each class signature in the specification's ``run`` command.
        """
        if any(isinstance(t, AssociationClass) for t in self.model.types):
            raise NotImplementedError("AlloyGeneratorNew does not support association classes yet")
        specification = BumlToAlloy(self.model, scope).translate()
        self.file_path = specification.write(self.build_generation_dir())

    def get_specification_path(self) -> str:
        """Returns the path to the generated Alloy specification file."""
        return self.file_path