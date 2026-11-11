import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from besser.BUML.metamodel.structural import DomainModel, Model

if TYPE_CHECKING:
    from besser.generators.regeneration import RegenerationReport

class InvalidIdentifierError(ValueError):
    """A model name that a generator would write into code is not a valid identifier."""


def require_valid_identifiers(model: DomainModel) -> None:
    """Refuse to render ``model`` while any name it emits as code is not an identifier.

    The ``NamedElement.name`` setter stays lenient so saved models keep loading;
    generators that write names verbatim into Python call this before rendering,
    so a name such as ``A=1;...`` can never become a statement in generated code.

    Raises:
        InvalidIdentifierError: With every offending name, one per line.
    """
    errors = model.invalid_identifier_names()
    if errors:
        raise InvalidIdentifierError("Cannot generate code from this model:\n" + "\n".join(errors))


# Interface for code generators
class GeneratorInterface(ABC):

    # Opt-in for regenerate(): the output must be byte-identical for one model, or a merge
    # could not tell a model change from noise (tests/generators/test_regenerable_generators.py).
    regenerable = False
    # The file, beside the generated entry point, that the generated code loads hand-written code
    # from and that no generation writes (``extensions.py`` of the backend); None when there is none.
    extension_file = None

    @abstractmethod
    def __init__(self, model: Model, output_dir: str = None):
        self.model = model
        self.output_dir = output_dir

    @abstractmethod
    def generate(self):
        pass

    def regenerate(self, adopt: bool = False) -> "RegenerationReport":
        """Bring the output folder up to this generator's model, keeping edits made in it.

        ``generate()`` overwrites; this runs it in a scratch folder and merges the result
        against what was generated last time, recorded in the folder. An edit is kept or
        flagged, never overwritten, and files the generator does not produce are left alone.
        Needs ``pip install "besser[regen]"`` and an explicit ``output_dir``.

        Args:
            adopt: Record a folder generated before this existed as this model's output,
                changing no file. Run it once, with the model those files came from.

        Returns:
            RegenerationReport: What changed, and what needs attention.

        Raises:
            RegenerationError: The folder cannot be regenerated safely; no generated file was changed.
        """
        from besser.generators.regeneration import regenerate_output

        return regenerate_output(self, adopt)

    def _regeneration_refusal(self) -> "str | None":
        """Why this instance cannot regenerate, as its options stand; ``None`` when it can."""
        return None

    def _keep_generated_values(self, previous: dict[str, bytes], generated: dict[str, bytes]) -> dict[str, bytes]:
        """``generated`` with any value that must stay the same across regenerations taken from ``previous``."""
        return generated

    def _without_secrets(self, files: dict[str, bytes]) -> dict[str, bytes]:
        """``files`` as the record holds them: it stays with the code, even once a secret has moved out of it."""
        return files

    @property
    def model(self) -> Model:
        return self.__model

    @model.setter
    def model(self, model: Model):
        self.__model = model

    @property
    def output_dir(self) -> str:
        return self.__output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str):
        self.__output_dir = output_dir

    def build_generation_path(self, file_name:str) -> str:
        file_path = os.path.join(self.build_generation_dir(), file_name)
        return file_path

    def build_generation_dir(self) -> str:
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir)
        else:
            working_path = os.path.abspath('')
            os.makedirs(os.path.join(working_path, "output"), exist_ok=True)
            file_path = os.path.join(working_path, "output")
        return file_path
