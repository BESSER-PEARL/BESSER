"""Helpers that read the editor JSON produced by the ``buml_to_json`` object converter."""


def object_names(result):
    """Object names in the editor JSON, keyed by element id."""
    return {i: e["name"] for i, e in result["elements"].items() if e["type"] == "ObjectName"}


def object_attributes(result):
    """Attribute values in the editor JSON, as ``"object.attribute = value"`` strings."""
    names = object_names(result)
    return {
        f"{names[e['owner']]}.{e['name']}"
        for e in result["elements"].values() if e["type"] == "ObjectAttribute"
    }


def object_links(result):
    """Links in the editor JSON, as ``"source -role-> target"`` strings."""
    names = object_names(result)
    return {
        f"{names[r['source']['element']]} -{r['name']}-> {names[r['target']['element']]}"
        for r in result["relationships"].values() if r["type"] == "ObjectLink"
    }
