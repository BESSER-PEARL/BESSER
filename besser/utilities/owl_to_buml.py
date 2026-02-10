from rdflib import Graph, Namespace, RDF, RDFS, OWL
from rdflib.term import URIRef, Literal
from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import process_class_diagram
from typing import Dict, List, Set, Tuple, Optional
import re


def get_local_name(uri: URIRef) -> str:
    """Extract the local name from a URI (e.g., 'Person' from 'http://example.org#Person')"""
    uri_str = str(uri)
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    elif '/' in uri_str:
        return uri_str.split('/')[-1]
    return uri_str


def map_xsd_to_buml_type(xsd_type: str) -> str:
    """Map XSD/OWL data types to BUML data types"""
    type_mapping = {
        'string': 'str',
        'int': 'int',
        'integer': 'int',
        'long': 'int',
        'short': 'int',
        'byte': 'int',
        'float': 'float',
        'double': 'float',
        'decimal': 'float',
        'boolean': 'bool',
        'date': 'date',
        'dateTime': 'datetime',
        'time': 'time',
        'anyURI': 'str',
        'nonNegativeInteger': 'int',
        'positiveInteger': 'int',
        'negativeInteger': 'int',
        'nonPositiveInteger': 'int',
    }
    
    # Extract the local name if it's a full URI
    if '#' in xsd_type or '/' in xsd_type:
        xsd_type = get_local_name(URIRef(xsd_type))
    
    return type_mapping.get(xsd_type.lower(), 'str')


def extract_cardinality(graph: Graph, property_uri: URIRef, class_uri: URIRef) -> Tuple[str, str]:
    """
    Extract cardinality constraints from OWL restrictions.
    Returns (min_cardinality, max_cardinality) as strings like '0', '1', '*'
    """
    min_card = '0'
    max_card = '*'
    
    # Check for restrictions on this property in this class
    for restriction in graph.subjects(RDF.type, OWL.Restriction):
        on_property = graph.value(restriction, OWL.onProperty)
        if on_property == property_uri:
            # Check if this restriction is used by the class
            for s, p, o in graph.triples((class_uri, RDFS.subClassOf, restriction)):
                # MinCardinality
                min_val = graph.value(restriction, OWL.minCardinality)
                if min_val is None:
                    min_val = graph.value(restriction, OWL.minQualifiedCardinality)
                if min_val:
                    min_card = str(min_val)
                
                # MaxCardinality
                max_val = graph.value(restriction, OWL.maxCardinality)
                if max_val is None:
                    max_val = graph.value(restriction, OWL.maxQualifiedCardinality)
                if max_val:
                    max_card = str(max_val) if int(max_val) > 1 else str(max_val)
                
                # Cardinality (exact)
                card_val = graph.value(restriction, OWL.cardinality)
                if card_val is None:
                    card_val = graph.value(restriction, OWL.qualifiedCardinality)
                if card_val:
                    min_card = max_card = str(card_val)
                
                # FunctionalProperty implies max cardinality of 1
                if (property_uri, RDF.type, OWL.FunctionalProperty) in graph:
                    max_card = '1'
                
                # InverseFunctionalProperty
                if (property_uri, RDF.type, OWL.InverseFunctionalProperty) in graph:
                    min_card = '1'
    
    # Check for functional properties
    if (property_uri, RDF.type, OWL.FunctionalProperty) in graph:
        max_card = '1'
    
    return min_card, max_card


def extract_owl_classes(graph: Graph) -> Set[URIRef]:
    """Extract all OWL classes from the ontology"""
    classes = set()
    
    # Get explicit owl:Class declarations
    for cls in graph.subjects(RDF.type, OWL.Class):
        classes.add(cls)
    
    # Get explicit rdfs:Class declarations
    for cls in graph.subjects(RDF.type, RDFS.Class):
        classes.add(cls)
    
    # Filter out OWL built-in classes
    owl_namespace = str(OWL)
    rdfs_namespace = str(RDFS)
    rdf_namespace = str(RDF)
    
    filtered_classes = {
        cls for cls in classes 
        if not (str(cls).startswith(owl_namespace) or 
                str(cls).startswith(rdfs_namespace) or
                str(cls).startswith(rdf_namespace))
    }
    
    return filtered_classes


def extract_datatype_properties(graph: Graph, class_uri: URIRef) -> List[Dict]:
    """Extract datatype properties (attributes) for a class"""
    attributes = []
    
    for prop in graph.subjects(RDF.type, OWL.DatatypeProperty):
        # Check if this property has the class as domain
        domains = list(graph.objects(prop, RDFS.domain))
        
        # If no domain specified, or if class is in domain
        if not domains or class_uri in domains:
            prop_name = get_local_name(prop)
            
            # Get range (data type)
            ranges = list(graph.objects(prop, RDFS.range))
            prop_type = 'str'  # default
            if ranges:
                prop_type = map_xsd_to_buml_type(str(ranges[0]))
            
            attributes.append({
                "name": prop_name,
                "type": prop_type,
                "visibility": "public"
            })
    
    return attributes


def extract_object_properties(graph: Graph, owl_classes: Set[URIRef]) -> List[Dict]:
    """Extract object properties (relationships) between classes"""
    relationships = []
    processed_pairs = set()  # To avoid duplicate relationships
    
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        prop_name = get_local_name(prop)
        
        # Get domain and range
        domains = list(graph.objects(prop, RDFS.domain))
        ranges = list(graph.objects(prop, RDFS.range))
        
        # Determine relationship type
        rel_type = "Association"
        
        # Check for subPropertyOf to determine composition/aggregation
        for super_prop in graph.objects(prop, RDFS.subPropertyOf):
            super_name = get_local_name(super_prop).lower()
            if 'part' in super_name or 'component' in super_name:
                rel_type = "Composition"
            elif 'member' in super_name or 'contains' in super_name:
                rel_type = "Aggregation"
        
        # If no explicit domain/range, try to infer from usage
        if not domains:
            domains = [s for s in graph.subjects(prop, None) if s in owl_classes]
        if not ranges:
            ranges = [o for s, p, o in graph.triples((None, prop, None)) if o in owl_classes]
        
        # Create relationships for all domain-range combinations
        for domain in domains:
            if domain not in owl_classes:
                continue
            
            for range_cls in ranges:
                if range_cls not in owl_classes:
                    continue
                
                source_name = get_local_name(domain)
                target_name = get_local_name(range_cls)
                
                # Avoid duplicate relationships
                pair_key = (source_name, target_name, prop_name)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Extract cardinality
                source_min, source_max = extract_cardinality(graph, prop, domain)
                target_min, target_max = extract_cardinality(graph, prop, range_cls)
                
                relationships.append({
                    "type": rel_type,
                    "source": source_name,
                    "target": target_name,
                    "sourceMultiplicity": f"{source_min}..{source_max}" if source_max != '*' else source_max,
                    "targetMultiplicity": f"{target_min}..{target_max}" if target_max != '*' else target_max,
                    "name": prop_name
                })
    
    return relationships


def extract_inheritance_relationships(graph: Graph, owl_classes: Set[URIRef]) -> List[Dict]:
    """Extract inheritance (subClassOf) relationships"""
    inheritance_rels = []
    
    for subclass in owl_classes:
        for superclass in graph.objects(subclass, RDFS.subClassOf):
            # Only process if superclass is also in our class set (not a restriction or built-in)
            if superclass in owl_classes:
                inheritance_rels.append({
                    "type": "Inheritance",
                    "source": get_local_name(subclass),
                    "target": get_local_name(superclass),
                    "sourceMultiplicity": "1",
                    "targetMultiplicity": "1",
                    "name": "inherits"
                })
    
    return inheritance_rels


def owl_to_buml(owl_path: str, model_name: str = "OWL_Imported_Model") -> DomainModel:
    """
    Convert an OWL ontology file to a B-UML DomainModel.
    
    This function parses an OWL ontology and extracts:
    - Classes (owl:Class)
    - Datatype properties as attributes
    - Object properties as associations
    - Inheritance relationships (rdfs:subClassOf)
    - Cardinality constraints from OWL restrictions
    
    Parameters
    ----------
    owl_path : str
        Path to the OWL ontology file (supports RDF/XML, Turtle, N3, etc.)
    model_name : str, optional
        Name for the generated B-UML model, by default "OWL_Imported_Model"
    
    Returns
    -------
    DomainModel
        The B-UML domain model object
    
    Raises
    ------
    RuntimeError
        If the OWL file cannot be parsed or processed
    
    Examples
    --------
    >>> domain_model = owl_to_buml("my_ontology.owl")
    >>> domain_model = owl_to_buml("university.ttl", "UniversityModel")
    """
    
    try:
        # Create RDF graph and parse OWL file
        graph = Graph()
        graph.parse(owl_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse OWL file at {owl_path}: {e}") from e
    
    try:
        # Extract classes
        owl_classes = extract_owl_classes(graph)
        
        if not owl_classes:
            raise ValueError("No OWL classes found in the ontology")
        
        # Build class specifications
        classes_spec = []
        
        for cls in owl_classes:
            class_name = get_local_name(cls)
            
            # Extract attributes (datatype properties)
            attributes = extract_datatype_properties(graph, cls)
            
            # Extract comments as documentation (optional)
            comments = list(graph.objects(cls, RDFS.comment))
            
            classes_spec.append({
                "className": class_name,
                "attributes": attributes,
                "methods": []  # OWL doesn't define methods
            })
        
        # Extract relationships (object properties)
        relationships = extract_object_properties(graph, owl_classes)
        
        # Extract inheritance relationships
        inheritance_rels = extract_inheritance_relationships(graph, owl_classes)
        relationships.extend(inheritance_rels)
        
        # Build the system specification
        system_spec = {
            "systemName": model_name,
            "classes": classes_spec,
            "relationships": relationships
        }
        
        # Convert to B-UML JSON format
        buml_json = convert_spec_json_to_buml(system_spec, title=model_name)
        
        if not buml_json:
            raise ValueError("Failed to convert OWL specification to B-UML format")
        
        # Convert to DomainModel
        domain_model: DomainModel = process_class_diagram(buml_json)
        
        return domain_model
        
    except Exception as e:
        raise RuntimeError(f"Failed to process OWL to B-UML conversion: {e}") from e


def convert_spec_json_to_buml(system_spec, title="OWL_Imported_Diagram"):
    """
    Convert class specifications from OWL-extracted JSON to Apollon/BUML format.
    
    """
    
    elements = {}
    relationships = {}
    
    # ID generation helpers
    def make_id(prefix, counter):
        return f"{prefix}-{counter}"
    
    class_map = {}
    class_counter = 1
    attr_counter = 1
    method_counter = 1
    rel_counter = 1
    
    # === Classes and their attributes/methods ===
    for cls in system_spec.get("classes", []):
        class_id = make_id("class", class_counter)
        class_map[cls["className"]] = class_id
        class_counter += 1
        
        # Create class element
        class_element = {
            "id": class_id,
            "type": "Class",
            "name": cls["className"],
            "attributes": [],
            "methods": []
        }
        
        # --- Attributes ---
        for attr in cls.get("attributes", []):
            attr_id = make_id("attr", attr_counter)
            attr_counter += 1
            
            visibility = attr.get("visibility", "public")
            visibility_symbol = (
                "+" if visibility == "public" else
                "-" if visibility == "private" else
                "#" if visibility == "protected" else
                "~"
            )
            name_str = f"{visibility_symbol} {attr['name']}: {attr['type']}"
            
            attr_element = {
                "id": attr_id,
                "type": "ClassAttribute",
                "owner": class_id,
                "name": name_str
            }
            
            class_element["attributes"].append(attr_id)
            elements[attr_id] = attr_element
        
        # --- Methods ---
        for method in cls.get("methods", []):
            method_id = make_id("method", method_counter)
            method_counter += 1
            
            visibility = method.get("visibility", "public")
            visibility_symbol = (
                "+" if visibility == "public" else
                "-" if visibility == "private" else
                "#" if visibility == "protected" else
                "~"
            )
            
            params = method.get("parameters", [])
            param_str = ", ".join([f"{p['name']}: {p['type']}" for p in params])
            
            return_type = method.get("returnType", "")
            if return_type.lower() == "void":
                return_type = ""
            
            name_str = f"{visibility_symbol} {method['name']}({param_str}): {return_type}"
            
            method_element = {
                "id": method_id,
                "type": "ClassMethod",
                "owner": class_id,
                "name": name_str
            }
            
            class_element["methods"].append(method_id)
            elements[method_id] = method_element
        
        elements[class_id] = class_element
    
    # === Relationships ===
    for rel in system_spec.get("relationships", []):
        rel_id = make_id("rel", rel_counter)
        rel_counter += 1
        
        rel_type = rel.get("type", "").lower()
        if rel_type in ["inheritance", "generalization"]:
            converted_type = "ClassInheritance"
        elif rel_type == "composition":
            converted_type = "ClassComposition"
        elif rel_type == "aggregation":
            converted_type = "ClassAggregation"
        else:
            converted_type = "ClassBidirectional"
        
        source_class_id = class_map.get(rel.get("sourceClass") or rel.get("source"))
        target_class_id = class_map.get(rel.get("targetClass") or rel.get("target"))
        
        if not source_class_id or not target_class_id:
            continue  # skip invalid relationships
        
        relationship_obj = {
            "id": rel_id,
            "type": converted_type,
            "source": {
                "element": source_class_id,
                "multiplicity": rel.get("sourceMultiplicity", "1"),
                "role": rel.get("sourceRole", "")
            },
            "target": {
                "element": target_class_id,
                "multiplicity": rel.get("targetMultiplicity", "1"),
                "role": rel.get("name", "")
            },
            "name": rel.get("name", "")
        }
        
        relationships[rel_id] = relationship_obj
    
    # === Final structure ===
    apollon_buml_json = {
        "title": title.replace(" ", "_"),
        "model": {
            "elements": elements,
            "relationships": relationships
        }
    }
    
    return apollon_buml_json