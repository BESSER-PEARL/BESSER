import functools
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Set

from besser.BUML.metamodel.structural import DomainModel, Model
from besser.utilities.generation_trace import (
    FileTrace, TraceRegion, TraceSource, extract_trace_regions, prefixed_traces, trace_run,
)

if TYPE_CHECKING:
    from besser.BUML.metamodel.traceability import LLM
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


def _one_run(generate: Callable) -> Callable:
    """``generate`` as one run: ``traced_files``, ``ai_files`` and ``ai_llm`` start empty, so they
    describe this run only, and its trace markers carry a nonce of their own (``trace_run``).

    A ``generate()`` that calls its base class's is part of the same run.
    """
    @functools.wraps(generate)
    def run(self, *args, **kwargs):
        if self.__dict__.get("_in_run"):
            return generate(self, *args, **kwargs)
        self.traced_files.clear()
        self.ai_files = set()
        self.ai_llm = None
        self._in_run = True
        try:
            with trace_run():
                return generate(self, *args, **kwargs)
        finally:
            self._in_run = False
    return run


class GeneratorInterface(ABC):
    """A code generator: ``generate()`` writes code derived from ``model`` into ``output_dir``.

    Tracing: ``traced_files`` maps each file the generator wrote with :meth:`_write_traced`
    (by path relative to its output directory, ``/``-separated) to the
    :class:`~besser.utilities.generation_trace.FileTrace` of what it was rendered from. It is
    always present and empty for a generator that records nothing, whose files still get
    file-level links (``besser.utilities.generation_trace.collect_generated_files``).
    ``ai_files`` are the files (same paths) an LLM wrote or changed during the run, with
    ``ai_llm`` the LLM that did: a trace records them with AI evidence, not a rule
    (``build_generation_trace(ai_files=..., llm=...)``). Both are empty for a deterministic run.
    """

    # Opt-in for regenerate(): the output must be byte-identical for one model, or a merge
    # could not tell a model change from noise (tests/generators/test_regenerable_generators.py).
    regenerable = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "generate" in cls.__dict__:
            cls.generate = _one_run(cls.__dict__["generate"])

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
    def traced_files(self) -> Dict[str, FileTrace]:
        """dict[str, FileTrace]: What each file the last run wrote was rendered from, by path relative to the
        output directory (``/``-separated). Empty until ``generate()`` writes a traced file, and
        emptied when it starts again."""
        # Created on first use: a subclass need not call this __init__ for its traces to be kept.
        if "_traced_files" not in self.__dict__:
            self._traced_files: Dict[str, FileTrace] = {}
        return self._traced_files

    @property
    def ai_files(self) -> Set[str]:
        """set[str]: The files, by path relative to the output directory, an LLM wrote or changed in the
        last run (a generator's optional LLM step). Empty for a deterministic run."""
        if "_ai_files" not in self.__dict__:
            self._ai_files: Set[str] = set()
        return self._ai_files

    @ai_files.setter
    def ai_files(self, paths: Set[str]):
        self._ai_files = set(paths)

    @property
    def ai_llm(self) -> Optional["LLM"]:
        """LLM: The LLM that wrote ``ai_files`` (None when there are none, or it is unknown)."""
        return self.__dict__.get("_ai_llm")

    @ai_llm.setter
    def ai_llm(self, llm: Optional["LLM"]):
        self._ai_llm = llm

    def _write_traced(self, relative_path: str, rendered: str, *sources: Optional[TraceSource],
                      output_dir: Optional[str] = None) -> str:
        """Write a template render without its trace markers, and record what it was rendered from.

        The file is written as UTF-8 with ``\\n`` line endings, whatever the platform, so its
        lines are the ones the trace counts. Folders on the way are created.

        Args:
            relative_path: Where to write, relative to ``output_dir``; also the key in ``traced_files``.
            rendered: The render of a template (of an environment from
                ``besser.utilities.generation_trace.traced_environment``), with its trace markers.
            *sources: The elements the whole file was generated from (``trace_source(element, rule)``);
                None entries (an element without an id) are skipped.
            output_dir: The folder the generator writes its output to; defaults to
                :meth:`build_generation_dir`. Pass it when the generator has its own default.

        Returns:
            str: The text written (the render without markers).
        """
        code, regions = extract_trace_regions(rendered)
        self._record_file(relative_path, code, regions, sources, output_dir)
        return code

    def _write_file(self, relative_path: str, text: str, *sources: Optional[TraceSource],
                    output_dir: Optional[str] = None) -> str:
        """Write text that is not a template render (a user's own config or skill) as :meth:`_write_traced` does.

        Nothing is extracted from it: it holds no trace markers, and what looks like one is
        written as it is. Only its whole-file ``sources`` are recorded.

        Returns:
            str: ``text``.
        """
        self._record_file(relative_path, text, (), sources, output_dir)
        return text

    def _record_file(self, relative_path: str, code: str, regions: Sequence[TraceRegion],
                     sources: Sequence[Optional[TraceSource]], output_dir: Optional[str]) -> None:
        base = os.path.realpath(output_dir if output_dir is not None else self.build_generation_dir())
        file_path = os.path.realpath(os.path.join(base, relative_path))
        try:
            inside = os.path.commonpath([base, file_path]) == base
        except ValueError:  # on different drives
            inside = False
        if not inside:
            raise ValueError(f"Refusing to write {relative_path!r}: it is outside the output folder.")
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, mode="w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        key = relative_path.replace(os.sep, "/")
        self.traced_files[key] = FileTrace(tuple(regions), tuple(s for s in sources if s is not None))

    def _merge_traced_files(self, child: "GeneratorInterface", prefix: str = "") -> None:
        """Add the traces of a generator this one ran into its output (a composite's part).

        A file the child's LLM wrote (``child.ai_files``) joins this generator's ``ai_files``
        instead, without its regions: they are not template output.

        Args:
            child: The generator, after its ``generate()``.
            prefix: The folder, relative to this generator's output directory, the child wrote
                into (``"backend"``); empty when it wrote into the same one.
        """
        ai_files = set(prefixed_traces(dict.fromkeys(child.ai_files), prefix))
        self.traced_files.update({path: trace for path, trace in prefixed_traces(child.traced_files, prefix).items()
                                  if path not in ai_files})
        if ai_files:
            self.ai_files.update(ai_files)
            self.ai_llm = child.ai_llm or self.ai_llm

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
