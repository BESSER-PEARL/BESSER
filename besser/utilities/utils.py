import pickle
import os
import datetime as _datetime
from typing import Any as _Any, Optional as _Optional

from besser.BUML.metamodel.structural import DomainModel


def timestamp_to_iso(timestamp: _Any) -> _Optional[str]:
    """Serialize an element ``timestamp`` as an ISO 8601 string.

    Returns ``None`` when the value is not a ``datetime`` (e.g. a timestamp
    that was never set), so callers can simply omit the field.
    """
    if isinstance(timestamp, _datetime.datetime):
        return timestamp.isoformat()
    return None


def explicit_timestamp_iso(element: _Any) -> _Optional[str]:
    """ISO 8601 form of ``element.timestamp`` when it was given explicitly.

    A timestamp that defaulted to the creation time (see
    ``Element.has_explicit_timestamp``) gives ``None``: writing it out would
    make serializing the same model twice produce different output.
    """
    if not getattr(element, "has_explicit_timestamp", False):
        return None
    return timestamp_to_iso(getattr(element, "timestamp", None))


def parse_iso_timestamp(value: _Any) -> _Optional[_datetime.datetime]:
    """Parse an ISO 8601 string written by :func:`timestamp_to_iso`.

    Tolerant by design: ``None``, non-string and malformed values return
    ``None`` so the caller falls back to the metamodel's default (the creation
    time). A timezone-aware value is converted to UTC and made naive, because
    the metamodel's default timestamps are naive and ``sort_by_timestamp``
    cannot compare naive with aware datetimes. UTC, not the server's local
    time, so the same input gives the same model on every host.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = _datetime.datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_datetime.timezone.utc).replace(tzinfo=None)
    return parsed


def sort_by_timestamp(obj_set: set) -> list:
    """
    Sorts a set of objects by their `timestamp` attribute.

    This function takes a set of objects (which are instances of the `NamedElement` class in BESSER)
    and returns a list of those objects, sorted in ascending order based on their `timestamp` attribute.

    Parameters:
    obj_set (set[NamedElement]): A set of objects to be sorted. Each object must have a `timestamp` attribute.

    Returns:
    list: A list of the objects sorted by their `timestamp` in ascending order.
    Ties (only possible with explicitly passed timestamps) are broken by name.
    """
    return sorted(obj_set, key=lambda x: (x.timestamp, str(getattr(x, "name", "") or "")))


class ModelSerializer():
    """
    ModelSerializer is a simple class for serializing and deserializing BESSER models and Python objects.

    Attributes:
        None

    Note:
        This class uses the `pickle` module for serialization and deserialization. Ensure that the models/objects you are
        serializing are safe and trusted since `pickle` can execute arbitrary code.
    """

    def __init__(self) -> None:
        pass

    def dump(self, model: any, output_dir: str=None, output_file_name: str = None):
        """Serialize and save a model to a file.

        Args:
            model (any): the B-UML model to be serialized and saved.
            output_dir (str, optional): the directory where the file should be saved. If not provided, the file will be saved in the current working directory.
            output_file_name (str, optional): The name of the output file. If not provided, a default name will be used based on the type of the model:
                For objects of type DomainModel: "{model_name}.buml"
                For other types: "model.pkl"

        Returns:
            None, but store the model as a file.
        """
        file_path = output_file_name
        if output_file_name is None:
            file_path = "model.pkl"
            if isinstance(model, DomainModel):
                file_path = model.name + ".buml"
        if output_dir is not None:
            file_path = os.path.join(output_dir, file_path)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
            f.close()

    def load(self, model_path: str):
        """Deserialize and load a model from a serialized file using pickle.

        Args:
            model_path (str): the path to the serialized model file.

        Returns:
            model_loaded: the deserialized model object.
        """
        with open(model_path, "rb") as f:
            model_loaded = pickle.load(f)
            f.close()
            return model_loaded
