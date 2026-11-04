from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.generators.alloy.alloy_specification import AlloyBounds, AlloySpecification
from besser.generators.alloy.alloy_translator import BumlToAlloy


class AlloyGenerator(GeneratorInterface):
    """Writes the Alloy specification of a B-UML domain model and its OCL invariants.

    ``model.als`` holds the specification built by :class:`BumlToAlloy`; the library modules
    it opens are written next to it.

    Raises:
        AlloyTranslationError: when the model, or one of its OCL invariants, cannot be translated.
    """

    def __init__(self, model: DomainModel, output_dir: str | None = None):
        super().__init__(model, output_dir)
        self.specification_model: AlloySpecification | None = None
        self.file_path = None

    def generate(self, scope: int = 5, seed: int = 0) -> None:
        """Writes the Alloy specification of the model to ``<output_dir>/model.als``.

        Args:
            scope: the scope of each class signature in the specification's ``run`` command.
            seed: the seed of the random dates filling the scope of ``Date``.
        """
        translator = BumlToAlloy(self.model, scope, seed)
        self.specification_model = translator.translate()
        self.file_path = self.specification_model.write(self.build_generation_dir())

    def get_specification_path(self) -> str:
        """Returns the path to the generated Alloy specification file."""
        return self.file_path

    def get_scopes(self) -> AlloyBounds | None:
        """Returns what the generated specification searches, ``None`` before :meth:`generate`."""
        return self.specification_model.get_scopes() if self.specification_model else None
