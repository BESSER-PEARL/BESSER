"""Traceability metamodel.

Records what happened to the artifacts of a BESSER project (model elements,
generated code) and the links derived between them. It follows W3C PROV:
an :class:`Artifact` is an entity, an :class:`Operation` is an activity and an
:class:`Actor` is an agent.

* An :class:`Operation` is performed by Actors with Tools and consumes and
  produces Artifacts. Its kind is its class: :class:`Modeling` edits models and
  holds the granular :class:`Change` records of what it did, :class:`CodeGeneration`
  (a :class:`Transformation`) derives code from models, :class:`Conversation` is
  a dialogue between a user and an agent. Whether a person or an agent did the
  work is told by its actors, not by its kind.
* A :class:`TraceLink` connects a source and a target Artifact (a class and the
  code generated from it). :class:`Evidence` says why the link holds and
  :class:`Confidence` how sure it is.

A trace is stored apart from the model it describes. It points at model
elements by their persistent ``Element.id`` (the editor element id), never by
object reference, so it outlives deleted elements and loads without the model.
Names are free text (a project owner is ``"BESSER User"``), so trace elements
inherit :class:`Element`, not :class:`NamedElement`; every one has an ``id``
(a UUID unless given) and a ``timestamp``.
"""

import uuid
from abc import ABC
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, List, Optional

from besser.BUML.metamodel.structural import Element, Model

__all__ = [
    "OperationStatus",
    "ChangeKind",
    "LogType",
    "TraceElement",
    "Artifact",
    "ModelArtifact",
    "CodeArtifact",
    "PromptArtifact",
    "Actor",
    "Person",
    "Agent",
    "Tool",
    "Editor",
    "CodeGenerator",
    "LLM",
    "LogRecord",
    "Change",
    "Operation",
    "Modeling",
    "Transformation",
    "CodeGeneration",
    "Conversation",
    "Evidence",
    "RuleEvidence",
    "AIEvidence",
    "ChangeEvidence",
    "Confidence",
    "TraceLink",
    "TraceModel",
]


class OperationStatus(Enum):
    """Lifecycle state of an operation."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class ChangeKind(Enum):
    """What a :class:`Change` did to its artifact."""
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


class LogType(Enum):
    """Kind of a :class:`LogRecord`."""
    INPUT_REQUEST = "INPUT_REQUEST"
    OUTPUT_RESPONSE = "OUTPUT_RESPONSE"
    INTERMEDIATE_MESSAGE = "INTERMEDIATE_MESSAGE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ERROR = "ERROR"
    INFO = "INFO"


def _text(value, owner: str, field: str) -> str:
    """``value`` when it is a non-blank str, else ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{field} must be a non-empty string.")
    return value


def _optional_text(value, owner: str, field: str) -> Optional[str]:
    """``value`` when it is None or a str, else ValueError."""
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{owner}.{field} must be a string or None.")
    return value


def _instance(value, expected_type, owner: str, field: str, optional: bool = False):
    """``value`` when it is an ``expected_type`` (or None when ``optional``), else TypeError."""
    if value is None and optional:
        return None
    if not isinstance(value, expected_type):
        raise TypeError(f"{owner}.{field} must be a {expected_type.__name__}, got {type(value).__name__}.")
    return value


def _instances(values: Optional[Iterable], expected_type, owner: str, field: str) -> list:
    """``values`` as a list without duplicates (first occurrence kept), each an ``expected_type``."""
    result = []
    for value in values or ():
        _instance(value, expected_type, owner, f"{field}[]")
        if not any(value is seen for seen in result):
            result.append(value)
    return result


class TraceElement(Element, ABC):
    """Superclass of every traceability element: an :class:`Element` that always has an id.

    Timestamps are naive UTC (an aware value is converted). A trace mixes times
    recorded by the browser and by the server, so unlike ``Element`` the
    default is the current UTC time, not the local one.

    Args:
        id (str): Persistent identifier (a new UUID when None).
        timestamp (datetime): When the element came to be (current UTC time when None).
    """

    def __init__(self, id: str = None, timestamp: datetime = None):
        super().__init__(timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
                         id=id if id is not None else str(uuid.uuid4()))
        _text(self.id, type(self).__name__, "id")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


class Artifact(TraceElement, ABC):
    """Something an operation consumes, produces or changes.

    Args:
        name (str): Display name, free text (an element name, a file path).
        parent (Artifact): The artifact this one is a fragment of (None for a whole artifact).
        id (str): Persistent identifier.
        timestamp (datetime): When the artifact was first recorded.
    """

    def __init__(self, name: str, parent: "Artifact" = None, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.name: str = name
        self.parent: Optional[Artifact] = parent

    @property
    def name(self) -> str:
        """str: Get the display name."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the display name.

        Raises:
            ValueError: If the name is not a non-empty string.
        """
        self.__name = _text(name, type(self).__name__, "name")

    @property
    def parent(self) -> Optional["Artifact"]:
        """Artifact: Get the artifact this one is a fragment of."""
        return self.__parent

    @parent.setter
    def parent(self, parent: Optional["Artifact"]):
        """Artifact: Set the artifact this one is a fragment of.

        Raises:
            TypeError: If parent is neither an Artifact nor None.
            ValueError: If the parent chain would contain this artifact.
        """
        _instance(parent, Artifact, type(self).__name__, "parent", optional=True)
        ancestor = parent
        while ancestor is not None:
            if ancestor is self:
                raise ValueError(f"Artifact {self.id!r} cannot be a fragment of itself.")
            ancestor = ancestor.parent
        self.__parent = parent


class ModelArtifact(Artifact):
    """A model, or an element of one.

    When the element is at hand (a trace loaded together with its model, as in a project
    exported to Python), the artifact references it (``element``) and takes its name, type
    and id from it instead of repeating them. Otherwise it records them: for an element
    that no longer exists (deleted, kept as history), or a trace stored apart from the
    model (the editor's JSON), where the element is known by its persistent id.

    Args:
        name (str): Last known name of the element (None when ``element`` is given).
        element_id (str): The ``Element.id`` of the element, the diagram id for a whole model
            (None when ``element`` is given and has no id yet).
        element_type (str): Its B-UML metaclass name: ``DomainModel``, ``Class``, ``Property``, ...
            (None when ``element`` is given).
        parent (Artifact): The model artifact holding this element.
        id (str): Persistent identifier of the artifact.
        timestamp (datetime): When the artifact was first recorded.
        element (Element): The referenced B-UML element, when it is at hand.

    Raises:
        ValueError: If neither ``element`` nor the recorded name, id and type are given.
    """

    def __init__(self, name: str = None, element_id: str = None, element_type: str = None, parent: Artifact = None,
                 id: str = None, timestamp: datetime = None, element: Element = None):
        self.element: Optional[Element] = element  # first: name, id and type fall back to it
        super().__init__(name, parent, id, timestamp)
        self.element_id: Optional[str] = element_id
        self.element_type: Optional[str] = element_type

    @property
    def element(self) -> Optional[Element]:
        """Element: Get the referenced B-UML element (None when only recorded)."""
        return self.__element

    @element.setter
    def element(self, element: Optional[Element]):
        """Element: Set the referenced B-UML element."""
        self.__element = _instance(element, Element, "ModelArtifact", "element", optional=True)

    @property
    def name(self) -> str:
        """str: Get the element's name: the referenced element's, else the recorded one."""
        return self.__element.name if self.__element is not None else self.__name

    @name.setter
    def name(self, name: Optional[str]):
        """str: Set the recorded name (optional when an element is referenced)."""
        self.__name = None if name is None and self.__element is not None else _text(name, "ModelArtifact", "name")

    @property
    def element_id(self) -> Optional[str]:
        """str: Get the element's persistent id: the recorded one, else the referenced element's."""
        if self.__element_id is not None:
            return self.__element_id
        return self.__element.id if self.__element is not None else None

    @element_id.setter
    def element_id(self, element_id: Optional[str]):
        """str: Set the recorded id (optional when an element is referenced).

        Raises:
            ValueError: If it is not a non-empty string while no element is referenced.
        """
        if element_id is None and self.__element is not None:
            self.__element_id = None
        else:
            self.__element_id = _text(element_id, "ModelArtifact", "element_id")

    @property
    def element_type(self) -> str:
        """str: Get the B-UML metaclass name: the referenced element's class, else the recorded one."""
        return type(self.__element).__name__ if self.__element is not None else self.__element_type

    @element_type.setter
    def element_type(self, element_type: Optional[str]):
        """str: Set the recorded metaclass name (optional when an element is referenced)."""
        if element_type is None and self.__element is not None:
            self.__element_type = None
        else:
            self.__element_type = _text(element_type, "ModelArtifact", "element_type")

    def resolve(self, model: Model) -> Optional[Element]:
        """The element of ``model`` this artifact stands for, or None when it is not there: the
        referenced element when it belongs to ``model``, else the one with this ``element_id``.

        A whole-model artifact resolves to ``model`` itself.
        """
        if self.__element is not None:
            if self.__element is model or any(e is self.__element for e in _model_elements(model)):
                return self.__element
            return None
        if model.id == self.element_id:
            return model
        for element in _model_elements(model):
            if element.id == self.element_id:
                return element
        return None


def _model_elements(model: Model) -> Iterable[Element]:
    """Every element of a domain model that can carry an editor id: types, their members, relations."""
    for type_ in getattr(model, "types", ()):
        yield type_
        yield from getattr(type_, "attributes", ())
        yield from getattr(type_, "methods", ())
        yield from getattr(type_, "literals", ())
    yield from getattr(model, "associations", ())
    yield from getattr(model, "generalizations", ())
    yield from getattr(model, "constraints", ())


class CodeArtifact(Artifact):
    """A generated or edited file, or a line range of one (a fragment).

    Args:
        name (str): Display name (the file name, or a label for the fragment).
        path (str): Path of the file, relative to the output root.
        content_hash (str): SHA-256 of the file content (of the fragment's lines for a fragment).
        content (str): The content itself, kept only below a size cap (None otherwise).
        start_line (int): First line of a fragment, 1-based (None for a whole file).
        end_line (int): Last line of a fragment, inclusive (None for a whole file).
        parent (Artifact): The file artifact holding this fragment.
        id (str): Persistent identifier.
        timestamp (datetime): When the artifact was first recorded.
    """

    def __init__(self, name: str, path: str, content_hash: str, content: str = None,
                 start_line: int = None, end_line: int = None, parent: Artifact = None,
                 id: str = None, timestamp: datetime = None):
        super().__init__(name, parent, id, timestamp)
        self.path: str = path
        self.content_hash: str = content_hash
        self.content: Optional[str] = content
        self.set_line_range(start_line, end_line)

    @property
    def path(self) -> str:
        """str: Get the path of the file."""
        return self.__path

    @path.setter
    def path(self, path: str):
        """str: Set the path of the file.

        Raises:
            ValueError: If it is not a non-empty string.
        """
        self.__path = _text(path, "CodeArtifact", "path")

    @property
    def content_hash(self) -> str:
        """str: Get the SHA-256 of the content."""
        return self.__content_hash

    @content_hash.setter
    def content_hash(self, content_hash: str):
        """str: Set the SHA-256 of the content.

        Raises:
            ValueError: If it is not a non-empty string.
        """
        self.__content_hash = _text(content_hash, "CodeArtifact", "content_hash")

    @property
    def content(self) -> Optional[str]:
        """str: Get the stored content (None when it was over the size cap)."""
        return self.__content

    @content.setter
    def content(self, content: Optional[str]):
        """str: Set the stored content.

        Raises:
            ValueError: If it is neither a string nor None.
        """
        self.__content = _optional_text(content, "CodeArtifact", "content")

    @property
    def start_line(self) -> Optional[int]:
        """int: Get the first line of the fragment (None for a whole file)."""
        return self.__start_line

    @property
    def end_line(self) -> Optional[int]:
        """int: Get the last line of the fragment (None for a whole file)."""
        return self.__end_line

    def set_line_range(self, start_line: Optional[int], end_line: Optional[int]):
        """Set both ends of the line range together, so they are checked against each other.

        Raises:
            ValueError: If only one end is given, or the range is not 1 <= start_line <= end_line.
        """
        if (start_line is None) != (end_line is None):
            raise ValueError("CodeArtifact needs both start_line and end_line, or neither.")
        if start_line is not None:
            if not all(isinstance(n, int) and not isinstance(n, bool) for n in (start_line, end_line)):
                raise ValueError("CodeArtifact line numbers must be integers.")
            if not 1 <= start_line <= end_line:
                raise ValueError(f"Invalid line range {start_line}-{end_line}: need 1 <= start_line <= end_line.")
        self.__start_line = start_line
        self.__end_line = end_line


class PromptArtifact(Artifact):
    """A message given to an agent, kept as an artifact because something was derived from it
    (a requirement, a model element): it is the source of a trace link. The other messages of a
    :class:`Conversation` stay in its log.

    Args:
        name (str): Display name (a short label for the prompt).
        content (str): The prompt text.
        content_hash (str): SHA-256 of the content.
        parent (Artifact): An artifact holding this one, if any.
        id (str): Persistent identifier.
        timestamp (datetime): When the prompt was given.
    """

    def __init__(self, name: str, content: str, content_hash: str, parent: Artifact = None,
                 id: str = None, timestamp: datetime = None):
        super().__init__(name, parent, id, timestamp)
        self.content: str = content
        self.content_hash: str = content_hash

    @property
    def content(self) -> str:
        """str: Get the prompt text."""
        return self.__content

    @content.setter
    def content(self, content: str):
        """str: Set the prompt text."""
        self.__content = _text(content, "PromptArtifact", "content")

    @property
    def content_hash(self) -> str:
        """str: Get the SHA-256 of the content."""
        return self.__content_hash

    @content_hash.setter
    def content_hash(self, content_hash: str):
        """str: Set the SHA-256 of the content."""
        self.__content_hash = _text(content_hash, "PromptArtifact", "content_hash")


class Actor(TraceElement, ABC):
    """Who performs an operation.

    Args:
        name (str): Display name, free text.
        id (str): Persistent identifier.
        timestamp (datetime): When the actor was first recorded.
    """

    def __init__(self, name: str, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.name: str = name

    @property
    def name(self) -> str:
        """str: Get the display name."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the display name.

        Raises:
            ValueError: If the name is not a non-empty string.
        """
        self.__name = _text(name, type(self).__name__, "name")


class Person(Actor):
    """A human actor (the project owner, or ``anonymous``)."""


class Agent(Actor):
    """A software actor acting on its own (the Spec-Driven Agent).

    Args:
        name (str): Display name.
        version (str): Version of the agent (None when unknown).
        id (str): Persistent identifier.
        timestamp (datetime): When the agent was first recorded.
    """

    def __init__(self, name: str, version: str = None, id: str = None, timestamp: datetime = None):
        super().__init__(name, id, timestamp)
        self.version: Optional[str] = version

    @property
    def version(self) -> Optional[str]:
        """str: Get the version of the agent."""
        return self.__version

    @version.setter
    def version(self, version: Optional[str]):
        """str: Set the version of the agent."""
        self.__version = _optional_text(version, "Agent", "version")


class Tool(TraceElement, ABC):
    """What an operation is carried out with.

    Args:
        name (str): Name of the tool (a generator key such as ``sqlalchemy``).
        version (str): Version of the tool (None when unknown).
        id (str): Persistent identifier.
        timestamp (datetime): When the tool was first recorded.
    """

    def __init__(self, name: str, version: str = None, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.name: str = name
        self.version: Optional[str] = version

    @property
    def name(self) -> str:
        """str: Get the name of the tool."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the tool.

        Raises:
            ValueError: If the name is not a non-empty string.
        """
        self.__name = _text(name, type(self).__name__, "name")

    @property
    def version(self) -> Optional[str]:
        """str: Get the version of the tool."""
        return self.__version

    @version.setter
    def version(self, version: Optional[str]):
        """str: Set the version of the tool."""
        self.__version = _optional_text(version, type(self).__name__, "version")


class Editor(Tool):
    """A modeling editor (the BESSER Web Modeling Editor)."""


class CodeGenerator(Tool):
    """A deterministic BESSER code generator."""


class LLM(Tool):
    """A large language model.

    Args:
        name (str): Model name.
        provider (str): Provider serving the model.
        version (str): Model version (None when unknown).
        id (str): Persistent identifier.
        timestamp (datetime): When the LLM was first recorded.
    """

    def __init__(self, name: str, provider: str, version: str = None, id: str = None, timestamp: datetime = None):
        super().__init__(name, version, id, timestamp)
        self.provider: str = provider

    @property
    def provider(self) -> str:
        """str: Get the provider serving the model."""
        return self.__provider

    @provider.setter
    def provider(self, provider: str):
        """str: Set the provider serving the model.

        Raises:
            ValueError: If it is not a non-empty string.
        """
        self.__provider = _text(provider, "LLM", "provider")


class LogRecord(TraceElement):
    """One logged message of an operation (a prompt, a tool call, an error).

    Args:
        content (str): The logged text.
        sequence_number (int): Position of the record within its operation.
        log_type (LogType): Kind of record.
        tool (Tool): The tool that produced or received it (None when not tool-specific).
        id (str): Persistent identifier.
        timestamp (datetime): When it was logged.
    """

    def __init__(self, content: str, sequence_number: int, log_type: LogType, tool: Tool = None,
                 id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.content: str = content
        self.sequence_number: int = sequence_number
        self.log_type: LogType = log_type
        self.tool: Optional[Tool] = tool

    @property
    def content(self) -> str:
        """str: Get the logged text."""
        return self.__content

    @content.setter
    def content(self, content: str):
        """str: Set the logged text.

        Raises:
            ValueError: If it is not a string.
        """
        if not isinstance(content, str):
            raise ValueError("LogRecord.content must be a string.")
        self.__content = content

    @property
    def sequence_number(self) -> int:
        """int: Get the position of the record within its operation."""
        return self.__sequence_number

    @sequence_number.setter
    def sequence_number(self, sequence_number: int):
        """int: Set the position of the record within its operation.

        Raises:
            ValueError: If it is not a non-negative integer.
        """
        if not isinstance(sequence_number, int) or isinstance(sequence_number, bool) or sequence_number < 0:
            raise ValueError("LogRecord.sequence_number must be a non-negative integer.")
        self.__sequence_number = sequence_number

    @property
    def log_type(self) -> LogType:
        """LogType: Get the kind of record."""
        return self.__log_type

    @log_type.setter
    def log_type(self, log_type: LogType):
        """LogType: Set the kind of record."""
        self.__log_type = _instance(log_type, LogType, "LogRecord", "log_type")

    @property
    def tool(self) -> Optional[Tool]:
        """Tool: Get the tool the record belongs to."""
        return self.__tool

    @tool.setter
    def tool(self, tool: Optional[Tool]):
        """Tool: Set the tool the record belongs to."""
        self.__tool = _instance(tool, Tool, "LogRecord", "tool", optional=True)


class Change(TraceElement):
    """One granular effect of an operation on one artifact.

    ``feature`` names the changed feature (``name``, ``type``, ``multiplicity``, ...)
    and is required for an UPDATED change and absent otherwise. ``new_value`` of a
    CREATED change and ``old_value`` of a DELETED one may hold the element's name.

    Args:
        kind (ChangeKind): Created, updated or deleted.
        artifact (Artifact): The artifact that changed.
        feature (str): The changed feature (UPDATED only).
        old_value (str): Value before the change.
        new_value (str): Value after the change.
        id (str): Persistent identifier.
        timestamp (datetime): When the change happened.
    """

    def __init__(self, kind: ChangeKind, artifact: Artifact, feature: str = None, old_value: str = None,
                 new_value: str = None, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.artifact: Artifact = artifact
        self.set_kind(kind, feature)
        self.old_value: Optional[str] = old_value
        self.new_value: Optional[str] = new_value

    @property
    def kind(self) -> ChangeKind:
        """ChangeKind: Get what the change did."""
        return self.__kind

    @property
    def feature(self) -> Optional[str]:
        """str: Get the changed feature (None unless UPDATED)."""
        return self.__feature

    def set_kind(self, kind: ChangeKind, feature: Optional[str]):
        """Set the kind and the changed feature together, so they are checked against each other.

        Raises:
            TypeError: If kind is not a ChangeKind.
            ValueError: If an UPDATED change has no feature, or another kind has one.
        """
        _instance(kind, ChangeKind, "Change", "kind")
        if kind is ChangeKind.UPDATED:
            _text(feature, "Change", "feature")
        elif feature is not None:
            raise ValueError(f"A {kind.value} change has no feature; got {feature!r}.")
        self.__kind = kind
        self.__feature = feature

    @property
    def artifact(self) -> Artifact:
        """Artifact: Get the artifact that changed."""
        return self.__artifact

    @artifact.setter
    def artifact(self, artifact: Artifact):
        """Artifact: Set the artifact that changed."""
        self.__artifact = _instance(artifact, Artifact, "Change", "artifact")

    @property
    def old_value(self) -> Optional[str]:
        """str: Get the value before the change."""
        return self.__old_value

    @old_value.setter
    def old_value(self, old_value: Optional[str]):
        """str: Set the value before the change."""
        self.__old_value = _optional_text(old_value, "Change", "old_value")

    @property
    def new_value(self) -> Optional[str]:
        """str: Get the value after the change."""
        return self.__new_value

    @new_value.setter
    def new_value(self, new_value: Optional[str]):
        """str: Set the value after the change."""
        self.__new_value = _optional_text(new_value, "Change", "new_value")


class Operation(TraceElement, ABC):
    """An activity on the project. Abstract: an operation is a :class:`Modeling`, a
    :class:`CodeGeneration` or a :class:`Conversation`.

    ``started_at`` is the element's timestamp.

    Args:
        performed_by (list[Actor]): Who performed it (at least one): a Person, an Agent, or both.
        status (OperationStatus): Lifecycle state (SUCCEEDED as default).
        started_at (datetime): When it started (current time when None).
        ended_at (datetime): When it ended (None while it runs).
        uses (list[Tool]): Tools it was carried out with.
        consumes (list[Artifact]): Artifacts it read.
        produces (list[Artifact]): Artifacts it created.
        logs (list[LogRecord]): Its log records, in order.
        id (str): Persistent identifier.

    Raises:
        TypeError: When instantiated directly, or as a bare Transformation.
    """

    def __init__(self, performed_by: List[Actor], status: OperationStatus = OperationStatus.SUCCEEDED,
                 started_at: datetime = None, ended_at: datetime = None, uses: List[Tool] = None,
                 consumes: List[Artifact] = None, produces: List[Artifact] = None, logs: List[LogRecord] = None,
                 id: str = None):
        if type(self) in (Operation, Transformation):
            raise TypeError(f"{type(self).__name__} is abstract: use Modeling, CodeGeneration or Conversation.")
        super().__init__(id, started_at)
        self.performed_by: List[Actor] = performed_by
        self.status: OperationStatus = status
        self.ended_at: Optional[datetime] = ended_at
        self.uses: List[Tool] = uses
        self.consumes: List[Artifact] = consumes
        self.produces: List[Artifact] = produces
        self.logs: List[LogRecord] = logs

    @property
    def performed_by(self) -> List[Actor]:
        """list[Actor]: Get who performed the operation."""
        return list(self.__performed_by)

    @performed_by.setter
    def performed_by(self, performed_by: List[Actor]):
        """list[Actor]: Set who performed the operation.

        Raises:
            ValueError: If no actor is given.
        """
        actors = _instances(performed_by, Actor, "Operation", "performed_by")
        if not actors:
            raise ValueError("An Operation is performed by at least one Actor.")
        self.__performed_by = actors

    @property
    def status(self) -> OperationStatus:
        """OperationStatus: Get the lifecycle state."""
        return self.__status

    @status.setter
    def status(self, status: OperationStatus):
        """OperationStatus: Set the lifecycle state."""
        self.__status = _instance(status, OperationStatus, "Operation", "status")

    @property
    def started_at(self) -> datetime:
        """datetime: Get when the operation started (its timestamp)."""
        return self.timestamp

    @property
    def ended_at(self) -> Optional[datetime]:
        """datetime: Get when the operation ended (None while it runs)."""
        return self.__ended_at

    @ended_at.setter
    def ended_at(self, ended_at: Optional[datetime]):
        """datetime: Set when the operation ended.

        Raises:
            ValueError: If it is before ``started_at``.
        """
        _instance(ended_at, datetime, "Operation", "ended_at", optional=True)
        if ended_at is not None and ended_at.tzinfo is not None:
            # Stored as naive UTC, as Element.timestamp is, so the two compare.
            ended_at = ended_at.astimezone(timezone.utc).replace(tzinfo=None)
        if ended_at is not None and ended_at < self.started_at:
            raise ValueError("Operation.ended_at cannot be before started_at.")
        self.__ended_at = ended_at

    @property
    def uses(self) -> List[Tool]:
        """list[Tool]: Get the tools the operation was carried out with."""
        return list(self.__uses)

    @uses.setter
    def uses(self, uses: List[Tool]):
        """list[Tool]: Set the tools the operation was carried out with."""
        self.__uses = _instances(uses, Tool, "Operation", "uses")

    @property
    def consumes(self) -> List[Artifact]:
        """list[Artifact]: Get the artifacts the operation read."""
        return list(self.__consumes)

    @consumes.setter
    def consumes(self, consumes: List[Artifact]):
        """list[Artifact]: Set the artifacts the operation read."""
        self.__consumes = _instances(consumes, Artifact, "Operation", "consumes")

    @property
    def produces(self) -> List[Artifact]:
        """list[Artifact]: Get the artifacts the operation created."""
        return list(self.__produces)

    @produces.setter
    def produces(self, produces: List[Artifact]):
        """list[Artifact]: Set the artifacts the operation created."""
        self.__produces = _instances(produces, Artifact, "Operation", "produces")

    @property
    def logs(self) -> List[LogRecord]:
        """list[LogRecord]: Get the operation's log records, in order."""
        return list(self.__logs)

    @logs.setter
    def logs(self, logs: List[LogRecord]):
        """list[LogRecord]: Set the operation's log records."""
        self.__logs = _instances(logs, LogRecord, "Operation", "logs")


class Modeling(Operation):
    """A modeling session: edits made to models, by a person, an agent (an
    assistant), or both, recorded change by change.

    Args:
        performed_by (list[Actor]): Who made the edits (at least one).
        changes (list[Change]): The edits, in order (at least one in a valid trace).
        **kwargs: The other :class:`Operation` arguments.
    """

    def __init__(self, performed_by: List[Actor], changes: List[Change] = None, **kwargs):
        super().__init__(performed_by, **kwargs)
        self.changes: List[Change] = changes

    @property
    def changes(self) -> List[Change]:
        """list[Change]: Get the session's edits, in order."""
        return list(self.__changes)

    @changes.setter
    def changes(self, changes: List[Change]):
        """list[Change]: Set the session's edits."""
        self.__changes = _instances(changes, Change, "Modeling", "changes")

    def add_change(self, change: Change):
        """Append an edit to the session."""
        self.changes = self.__changes + [change]


class Transformation(Operation, ABC):
    """An operation that derives new artifacts from others (and the trace links
    between them) without editing anything. Abstract: see :class:`CodeGeneration`."""


class CodeGeneration(Transformation):
    """A generator run: code derived from models.

    It uses at least one :class:`CodeGenerator` (checked by ``TraceModel.validate``),
    consumes the model artifacts and produces the code artifacts it links them to.
    """


class Conversation(Operation):
    """A dialogue between a user and an agent (the BESSER assistant), with its messages in the log.

    What it leads to varies, so it is not a :class:`Transformation`: a clarification question
    produces nothing; stated requirements are artifacts it produces, linked from the
    :class:`PromptArtifact` they came from; an edit or a generation it asks for is a separate
    :class:`Modeling` or :class:`CodeGeneration`. At least one of its actors is an :class:`Agent`
    (checked by ``TraceModel.validate``).
    """


class Evidence(TraceElement, ABC):
    """Why a trace link holds. Its timestamp is when it was recorded."""


class RuleEvidence(Evidence):
    """The link was made by a deterministic rule (a generator template block).

    Args:
        rule (str): The rule, e.g. ``sql_alchemy_template.py.j2#class``.
        id (str): Persistent identifier.
        timestamp (datetime): When it was recorded.
    """

    def __init__(self, rule: str, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.rule: str = rule

    @property
    def rule(self) -> str:
        """str: Get the rule that made the link."""
        return self.__rule

    @rule.setter
    def rule(self, rule: str):
        """str: Set the rule that made the link."""
        self.__rule = _text(rule, "RuleEvidence", "rule")


class AIEvidence(Evidence):
    """The link was inferred by an AI technique, supported by log records (prompts, responses).

    Args:
        algorithm (str): The technique or model that inferred the link.
        logs (list[LogRecord]): The log records it is based on.
        id (str): Persistent identifier.
        timestamp (datetime): When it was recorded.
    """

    def __init__(self, algorithm: str, logs: List[LogRecord] = None, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.algorithm: str = algorithm
        self.logs: List[LogRecord] = logs

    @property
    def algorithm(self) -> str:
        """str: Get the technique that inferred the link."""
        return self.__algorithm

    @algorithm.setter
    def algorithm(self, algorithm: str):
        """str: Set the technique that inferred the link."""
        self.__algorithm = _text(algorithm, "AIEvidence", "algorithm")

    @property
    def logs(self) -> List[LogRecord]:
        """list[LogRecord]: Get the log records the inference is based on."""
        return list(self.__logs)

    @logs.setter
    def logs(self, logs: List[LogRecord]):
        """list[LogRecord]: Set the log records the inference is based on."""
        self.__logs = _instances(logs, LogRecord, "AIEvidence", "logs")


class ChangeEvidence(Evidence):
    """The link was inferred from changes (e.g. elements always edited together).

    Args:
        changes (list[Change]): The changes it is based on (at least one).
        id (str): Persistent identifier.
        timestamp (datetime): When it was recorded.
    """

    def __init__(self, changes: List[Change], id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.changes: List[Change] = changes

    @property
    def changes(self) -> List[Change]:
        """list[Change]: Get the changes the inference is based on."""
        return list(self.__changes)

    @changes.setter
    def changes(self, changes: List[Change]):
        """list[Change]: Set the changes the inference is based on.

        Raises:
            ValueError: If no change is given.
        """
        values = _instances(changes, Change, "ChangeEvidence", "changes")
        if not values:
            raise ValueError("ChangeEvidence is based on at least one Change.")
        self.__changes = values


class Confidence(TraceElement):
    """How sure a trace link is, as a probability. Its timestamp is when it was calculated.

    Args:
        value (float): Probability between 0 and 1.
        based_on (list[Evidence]): The evidence the value is calculated from.
        explanation (str): How the value was obtained (None when self-evident).
        id (str): Persistent identifier.
        timestamp (datetime): When it was calculated.
    """

    def __init__(self, value: float, based_on: List[Evidence] = None, explanation: str = None,
                 id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.value: float = value
        self.based_on: List[Evidence] = based_on
        self.explanation: Optional[str] = explanation

    @property
    def value(self) -> float:
        """float: Get the probability."""
        return self.__value

    @value.setter
    def value(self, value: float):
        """float: Set the probability.

        Raises:
            ValueError: If it is not a number between 0 and 1 inclusive.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("Confidence.value must be a probability between 0 and 1 inclusive.")
        self.__value = float(value)

    @property
    def based_on(self) -> List[Evidence]:
        """list[Evidence]: Get the evidence the value is calculated from."""
        return list(self.__based_on)

    @based_on.setter
    def based_on(self, based_on: List[Evidence]):
        """list[Evidence]: Set the evidence the value is calculated from."""
        self.__based_on = _instances(based_on, Evidence, "Confidence", "based_on")

    @property
    def explanation(self) -> Optional[str]:
        """str: Get how the value was obtained."""
        return self.__explanation

    @explanation.setter
    def explanation(self, explanation: Optional[str]):
        """str: Set how the value was obtained."""
        self.__explanation = _optional_text(explanation, "Confidence", "explanation")


class TraceLink(TraceElement):
    """A directed link from a source artifact to a target artifact, made by an operation.

    Its timestamp is when the link was created. It is *suspect* when its source
    changed after that; see :meth:`TraceModel.is_suspect`.

    Args:
        source (Artifact): Where the link starts (a class).
        target (Artifact): Where it ends (the code generated from the class).
        operation (Operation): The operation that made it.
        evidence (list[Evidence]): Why it holds.
        confidence (Confidence): How sure it is (None when not assessed).
        id (str): Persistent identifier.
        timestamp (datetime): When it was created.
    """

    def __init__(self, source: Artifact, target: Artifact, operation: Operation, evidence: List[Evidence] = None,
                 confidence: Confidence = None, id: str = None, timestamp: datetime = None):
        super().__init__(id, timestamp)
        self.set_ends(source, target)
        self.operation: Operation = operation
        self.evidence: List[Evidence] = evidence
        self.confidence: Optional[Confidence] = confidence

    @property
    def source(self) -> Artifact:
        """Artifact: Get where the link starts."""
        return self.__source

    @property
    def target(self) -> Artifact:
        """Artifact: Get where the link ends."""
        return self.__target

    def set_ends(self, source: Artifact, target: Artifact):
        """Set both ends together, so they are checked against each other.

        Raises:
            TypeError: If an end is not an Artifact.
            ValueError: If source and target are the same artifact.
        """
        _instance(source, Artifact, "TraceLink", "source")
        _instance(target, Artifact, "TraceLink", "target")
        if source is target:
            raise ValueError("A TraceLink cannot link an artifact to itself.")
        self.__source = source
        self.__target = target

    @property
    def operation(self) -> Operation:
        """Operation: Get the operation that made the link."""
        return self.__operation

    @operation.setter
    def operation(self, operation: Operation):
        """Operation: Set the operation that made the link."""
        self.__operation = _instance(operation, Operation, "TraceLink", "operation")

    @property
    def evidence(self) -> List[Evidence]:
        """list[Evidence]: Get why the link holds."""
        return list(self.__evidence)

    @evidence.setter
    def evidence(self, evidence: List[Evidence]):
        """list[Evidence]: Set why the link holds."""
        self.__evidence = _instances(evidence, Evidence, "TraceLink", "evidence")

    @property
    def confidence(self) -> Optional[Confidence]:
        """Confidence: Get how sure the link is."""
        return self.__confidence

    @confidence.setter
    def confidence(self, confidence: Optional[Confidence]):
        """Confidence: Set how sure the link is."""
        self.__confidence = _instance(confidence, Confidence, "TraceLink", "confidence", optional=True)


class TraceModel(Model):
    """The root of a trace: everything recorded for one project, stored apart from its models.

    Args:
        name (str): Python-identifier-safe name (``NamedElement.name`` rules apply).
        project_id (str): Id of the project the trace belongs to (None for a fragment
            not yet merged into a project trace, such as one returned by a generation).
        artifacts (list[Artifact]): Every artifact recorded.
        operations (list[Operation]): Every operation, in order.
        trace_links (list[TraceLink]): Every trace link.
        actors (list[Actor]): Every actor.
        tools (list[Tool]): Every tool.
    """

    def __init__(self, name: str = "trace_model", project_id: str = None, artifacts: List[Artifact] = None,
                 operations: List[Operation] = None, trace_links: List[TraceLink] = None,
                 actors: List[Actor] = None, tools: List[Tool] = None):
        super().__init__(name)
        self.project_id: Optional[str] = project_id
        self.artifacts: List[Artifact] = artifacts
        self.operations: List[Operation] = operations
        self.trace_links: List[TraceLink] = trace_links
        self.actors: List[Actor] = actors
        self.tools: List[Tool] = tools

    @property
    def project_id(self) -> Optional[str]:
        """str: Get the id of the project the trace belongs to."""
        return self.__project_id

    @project_id.setter
    def project_id(self, project_id: Optional[str]):
        """str: Set the id of the project the trace belongs to."""
        self.__project_id = _optional_text(project_id, "TraceModel", "project_id")

    @property
    def artifacts(self) -> List[Artifact]:
        """list[Artifact]: Get every artifact recorded."""
        return list(self.__artifacts)

    @artifacts.setter
    def artifacts(self, artifacts: List[Artifact]):
        """list[Artifact]: Set every artifact recorded."""
        self.__artifacts = _instances(artifacts, Artifact, "TraceModel", "artifacts")

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Add an artifact and return it."""
        self.artifacts = self.__artifacts + [artifact]
        return artifact

    @property
    def operations(self) -> List[Operation]:
        """list[Operation]: Get every operation, in order."""
        return list(self.__operations)

    @operations.setter
    def operations(self, operations: List[Operation]):
        """list[Operation]: Set every operation."""
        self.__operations = _instances(operations, Operation, "TraceModel", "operations")

    def add_operation(self, operation: Operation) -> Operation:
        """Add an operation and return it."""
        self.operations = self.__operations + [operation]
        return operation

    @property
    def trace_links(self) -> List[TraceLink]:
        """list[TraceLink]: Get every trace link."""
        return list(self.__trace_links)

    @trace_links.setter
    def trace_links(self, trace_links: List[TraceLink]):
        """list[TraceLink]: Set every trace link."""
        self.__trace_links = _instances(trace_links, TraceLink, "TraceModel", "trace_links")

    def add_trace_link(self, trace_link: TraceLink) -> TraceLink:
        """Add a trace link and return it."""
        self.trace_links = self.__trace_links + [trace_link]
        return trace_link

    @property
    def actors(self) -> List[Actor]:
        """list[Actor]: Get every actor."""
        return list(self.__actors)

    @actors.setter
    def actors(self, actors: List[Actor]):
        """list[Actor]: Set every actor."""
        self.__actors = _instances(actors, Actor, "TraceModel", "actors")

    def add_actor(self, actor: Actor) -> Actor:
        """Add an actor and return it."""
        self.actors = self.__actors + [actor]
        return actor

    @property
    def tools(self) -> List[Tool]:
        """list[Tool]: Get every tool."""
        return list(self.__tools)

    @tools.setter
    def tools(self, tools: List[Tool]):
        """list[Tool]: Set every tool."""
        self.__tools = _instances(tools, Tool, "TraceModel", "tools")

    def add_tool(self, tool: Tool) -> Tool:
        """Add a tool and return it."""
        self.tools = self.__tools + [tool]
        return tool

    def fragments_of(self, artifact: Artifact) -> List[Artifact]:
        """The artifacts whose parent is ``artifact``."""
        return [a for a in self.__artifacts if a.parent is artifact]

    def model_artifact_for(self, element_id: str) -> Optional[ModelArtifact]:
        """The model artifact referencing ``element_id``, or None."""
        return next((a for a in self.__artifacts
                     if isinstance(a, ModelArtifact) and a.element_id == element_id), None)

    def changes_of(self, artifact: Artifact) -> List[Change]:
        """Every change recorded on ``artifact``, in time order."""
        changes = [c for op in self.__operations if isinstance(op, Modeling) for c in op.changes
                   if c.artifact is artifact]
        return sorted(changes, key=lambda c: c.timestamp)

    def is_suspect(self, trace_link: TraceLink) -> bool:
        """True when the link's source, or a fragment of it, changed after the link was made:
        the target may no longer match it."""
        pending = [trace_link.source]
        while pending:
            artifact = pending.pop()
            if any(c.timestamp > trace_link.timestamp for c in self.changes_of(artifact)):
                return True
            pending.extend(self.fragments_of(artifact))
        return False

    def is_stale(self, trace_link: TraceLink) -> bool:
        """``TraceLink./stale``, derived: the link may no longer hold, because it is suspect or
        because a link into its source, further upstream, is stale."""
        incoming: dict[int, List[TraceLink]] = {}
        for link in self.__trace_links:
            incoming.setdefault(id(link.target), []).append(link)
        stale: dict[int, bool] = {}

        def visit(link: TraceLink) -> bool:
            if id(link) not in stale:
                stale[id(link)] = False  # a cycle back to this link adds nothing
                stale[id(link)] = self.is_suspect(link) or any(
                    visit(upstream) for upstream in incoming.get(id(link.source), []))
            return stale[id(link)]

        return visit(trace_link)

    def validate(self, raise_exception: bool = True) -> dict:
        """Check that every reference stays inside the trace and every id is unique.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: Validation result with success flag, errors, and warnings.
        """
        errors: list[str] = []
        artifacts = set(map(id, self.__artifacts))
        actors = set(map(id, self.__actors))
        tools = set(map(id, self.__tools))
        operations = set(map(id, self.__operations))

        def check(element, collection: set, what: str, where: str):
            if element is not None and id(element) not in collection:
                errors.append(f"{where} references {what} {element.id!r}, which is not in the trace.")

        for artifact in self.__artifacts:
            check(artifact.parent, artifacts, "artifact", f"Artifact {artifact.id!r} (parent)")
        producer: dict[int, str] = {}
        for operation in self.__operations:
            where = f"Operation {operation.id!r}"
            for actor in operation.performed_by:
                check(actor, actors, "actor", where)
            for tool in operation.uses:
                check(tool, tools, "tool", where)
            for artifact in operation.consumes + operation.produces:
                check(artifact, artifacts, "artifact", where)
            for artifact in operation.produces:
                if id(artifact) in producer:
                    errors.append(f"Artifact {artifact.id!r} is produced by both operation "
                                  f"{producer[id(artifact)]!r} and {operation.id!r}.")
                producer[id(artifact)] = operation.id
            for log in operation.logs:
                check(log.tool, tools, "tool", f"LogRecord {log.id!r}")
            if isinstance(operation, Modeling):
                if not operation.changes:
                    errors.append(f"{where} is a Modeling with no changes.")
                for change in operation.changes:
                    check(change.artifact, artifacts, "artifact", f"Change {change.id!r}")
            elif isinstance(operation, CodeGeneration):
                if not any(isinstance(tool, CodeGenerator) for tool in operation.uses):
                    errors.append(f"{where} is a CodeGeneration that uses no CodeGenerator.")
            elif isinstance(operation, Conversation):
                if not any(isinstance(actor, Agent) for actor in operation.performed_by):
                    errors.append(f"{where} is a Conversation with no Agent among its actors.")
        for link in self.__trace_links:
            where = f"TraceLink {link.id!r}"
            check(link.source, artifacts, "artifact", where)
            check(link.target, artifacts, "artifact", where)
            check(link.operation, operations, "operation", where)

        seen: dict[str, str] = {}
        for element in self._all_elements():
            kind = type(element).__name__
            if element.id in seen:
                errors.append(f"Id {element.id!r} is used by both a {seen[element.id]} and a {kind}.")
            seen[element.id] = kind

        result = {"success": not errors, "errors": errors, "warnings": []}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _all_elements(self) -> List[TraceElement]:
        """Every trace element owned by the model, each once."""
        elements: list = self.artifacts + self.actors + self.tools + self.operations
        for operation in self.__operations:
            elements += operation.logs
            if isinstance(operation, Modeling):
                elements += operation.changes
        for link in self.__trace_links:
            elements.append(link)
            elements += link.evidence
            if link.confidence is not None:
                elements.append(link.confidence)
        return elements
