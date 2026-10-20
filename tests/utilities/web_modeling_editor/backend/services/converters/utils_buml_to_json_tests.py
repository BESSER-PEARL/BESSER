"""Helpers that read the v4 editor JSON produced by the ``buml_to_json`` object converter."""


def object_names(result):
    """Object names in the editor JSON, keyed by node id."""
    return {n["id"]: n["data"]["name"] for n in result["nodes"] if n["type"] == "objectName"}


def object_attributes(result):
    """Attribute values in the editor JSON, as ``"object.attribute = value"`` strings."""
    return {
        f"{n['data']['name']}.{row['name']} = {row['value']}"
        for n in result["nodes"] if n["type"] == "objectName"
        for row in n["data"]["attributes"]
    }


def object_links(result):
    """Links in the editor JSON, as ``"source -role-> target"`` strings."""
    names = object_names(result)
    return {
        f"{names[e['source']]} -{e['data']['name']}-> {names[e['target']]}"
        for e in result["edges"] if e["type"] == "ObjectLink"
    }
