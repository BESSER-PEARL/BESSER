import json
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.object.object import DataValue
from besser.generators import GeneratorInterface


class JSONObjectGenerator(GeneratorInterface):
    """Generate JSON documents from BESSER ObjectModel instances."""

    def __init__(self, model: ObjectModel, output_dir: str = None):
        if not isinstance(model, ObjectModel):
            raise TypeError("JSONObjectGenerator expects an ObjectModel instance")
        super().__init__(model, output_dir)

    def generate(self):
        payload = self._build_document()
        file_name = f"{self._sanitize_name(self.model.name or 'object_model')}.json"
        file_path = self.build_generation_path(file_name)
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2, ensure_ascii=False)

    def _build_document(self) -> Dict[str, Any]:
        objects = sorted(self.model.objects, key=lambda obj: obj.name_.lower())
        relationships = self._build_relationship_index(objects)

        items = []
        for obj in objects:
            entry: Dict[str, Any] = {
                "id": obj.name_,
                "class": obj.classifier.name,
            }

            attributes = self._serialize_attributes(obj)
            if attributes:
                entry["attributes"] = attributes

            obj_relationships = relationships.get(obj.name_)
            if obj_relationships:
                entry["relationships"] = obj_relationships

            items.append(entry)

        document: Dict[str, Any] = {
            "name": self.model.name,
            "objects": items,
        }

        description = getattr(getattr(self.model, "metadata", None), "description", None)
        if description:
            document["description"] = description

        return document

    def _serialize_attributes(self, obj) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for slot in getattr(obj, "slots", []) or []:
            attribute = getattr(slot, "attribute", None)
            if not attribute:
                continue
            value = getattr(slot, "value", None)
            data[attribute.name] = self._serialize_value(value)
        return data

    def _build_relationship_index(self, objects) -> Dict[str, Dict[str, List[str]]]:
        rel_index: Dict[str, Dict[str, List[str]]] = {obj.name_: {} for obj in objects}
        link_set = getattr(self.model, "links", set()) or []

        for link in link_set:
            association_name = getattr(getattr(link, "association", None), "name", None)
            connections = getattr(link, "connections", []) or []
            for idx, end in enumerate(connections):
                source_obj = getattr(end, "object", None)
                if not source_obj:
                    continue
                target_names = []
                for other_idx, other_end in enumerate(connections):
                    if other_idx == idx:
                        continue
                    target_obj = getattr(other_end, "object", None)
                    if target_obj:
                        target_names.append(target_obj.name_)

                if not target_names:
                    continue

                rel_name = getattr(getattr(end, "association_end", None), "name", None) or association_name
                if not rel_name:
                    rel_name = "association"

                existing = rel_index.setdefault(source_obj.name_, {}).setdefault(rel_name, [])
                for target in target_names:
                    if target not in existing:
                        existing.append(target)

        return {key: value for key, value in rel_index.items() if value}

    def _serialize_value(self, slot_value):
        value = slot_value
        if isinstance(slot_value, DataValue):
            value = slot_value.value

        literal_name = getattr(value, "name", None)
        if literal_name and value.__class__.__name__ == "EnumerationLiteral":
            return literal_name

        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, set):
            return [self._serialize_value(item) for item in sorted(value, key=lambda item: str(item))]
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]
        if hasattr(value, "name_"):
            return value.name_
        return value

    def _sanitize_name(self, name: str) -> str:
        cleaned = (name or "object_model").strip().replace(" ", "_")
        return cleaned or "object_model"
