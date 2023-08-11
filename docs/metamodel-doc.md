# BUML Modeling

BUML is based on UML and allows you to define models using Python Object-Oriented Programming. The main classes, their attributes, and basic examples are described below.

## Main Elements

* [Property](metamodel-doc.md#property)
* [Class](metamodel-doc.md#class)
* [Association](metamodel-doc.md#association)
* [BinaryAssociation](metamodel-doc.md#binary-association)
* [AssocationClass](metamodel-doc.md#association-class)
* [Generalization](metamodel-doc.md#generalization)
* [GeneralizationSet](metamodel-doc.md#generalization-set)
* [Package](metamodel-doc.md#package)
* [Constraint](metamodel-doc.md#constraint)
* [DomainModel](metamodel-doc.md#domain-model)

## Property
> Property(name, owner, property_type, multiplicity, visibility, is_composite, is_navigable, is_aggregation)

A Property object can be used to specify an *attribute* of a [Class](metamodel-doc.md#class) or an *end* of an [Association](metamodel-doc.md#association).

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| name | str | Name of the Property | |
| owner | Type | Owner of the property. Should be a [Class](metamodel-doc.md#class) if the Property describes a class attribute | |
| property_type | Type | Type of property. It could be a *PrimitiveDataType* if the property is to define an attribute of a [Class](metamodel-doc.md#class), or it could be a [Class](metamodel-doc.md#class) if the property defines an *end* of an [Association](metamodel-doc.md#association) | |
| multiplicity | Multiplicity | Multiplicity of the Property when it define an *end* of an [Association](metamodel-doc.md#association) |   (1,1)   |
| visibility | str | The visibility of the property can be *public*, *private*, *protected*, or *package* | *public* |
| is_composite | bool | When the Property defines an *end* of an [Association](metamodel-doc.md#association), *is_composite* specifies whether the relationship is a composite or not | False |
| is_navigable | bool | Specifies the navigability between *ends* of an [Association](metamodel-doc.md#association) | True |
| is_aggregation | bool | When the Property defines an *end* of an [Association](metamodel-doc.md#association), *is_aggregation* specifies whether the relationship is an aggregation or not    | False |

### Example
The following example defines two *attributes*.

```python
# Import BUML classes
from metamodel.structural.structural import Property, PrimitiveDataType, Multiplicity

# Attributes definition
attribute1: Property = Property(name="attr1", owner=None, property_type=PrimitiveDataType("int"))
attribute2: Property = Property(name="attr2", owner=None, property_type=PrimitiveDataType("str"))
```

## Class
> Class(name, attributes, is_abstract)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| name | str | Name of the Class | |
| attributes | set([Property](metamodel-doc.md#property)) | Set of *attributes* of the class | |
| is_abstract | Bool | Set to *True* if the Class is abstract, *False* otherwise | False |

### Example

<img src="/docs/img/class.jpg" alt="Class" style="height: 20%; width:20%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Property, PrimitiveDataType, Multiplicity, Class

# Attributes definition
attribute1: Property = Property(name="attr1", owner=None, property_type=PrimitiveDataType("int"))
attribute2: Property = Property(name="attr2", owner=None, property_type=PrimitiveDataType("str"))

# Class definition
class1: Class = Class(name="Cls1", attributes={attribute1, attribute2})
```

## Association
> Association(name, ends)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| name | str | Name of the Association | |
| ends | set([Property](metamodel-doc.md#property)) | Set of *ends* of the Association | |

### Example

<img src="/docs/img/association.jpg" alt="Association" style="height: 55%; width:55%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Multiplicity, Property, Association

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)
class3: Class = Class(name="Cls3", attributes=None)

# Ends definition
aend1: Property = Property(name="end1", owner=None, property_type=class1, multiplicity=Multiplicity(1, 1), is_composite=True)
aend2: Property = Property(name="end2", owner=None, property_type=class2, multiplicity=Multiplicity(0, "*"))
aend3: Property = Property(name="end3", owner=None, property_type=class3, multiplicity=Multiplicity(0, 1))

# Association definition
association: Association = Association(name="association", ends={aend1, aend2,aend3})
```

## BinaryAssociation

> BinaryAssociation(name, ends)

BinaryAssociation is similar to [Association](metamodel-doc.md#association) but with the constraint that it can only have two ends.

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| name | str | Name of the BinaryAssociation | |
| ends | set([Property](metamodel-doc.md#property)) | Set of *ends* (a binary must have exactly two ends) | |

### Example

<img src="/docs/img/binaryAsso.jpg" alt="Binary Association" style="height: 60%; width:60%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Multiplicity, Property, BinaryAssociation

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)

# Ends definition
aend1: Property = Property(name="end1", owner=None, property_type=class1, multiplicity=Multiplicity(1, 1))
aend2: Property = Property(name="end2", owner=None, property_type=class2, multiplicity=Multiplicity(0, "*"))

# BinaryAssociation definition
binaryA: BinaryAssociation = BinaryAssociation(name="BinaryA1", ends={aend1, aend2})
```

## AssocationClass
> AssociationClass(name, attributes, association)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| name | str | Name of the AssociationClass | |
| attributes | set([Property](metamodel-doc.md#property)) | Set of *attributes* of the [Class](metamodel-doc.md#class) | |
| association | [Association](metamodel-doc.md#association) | It could be an [Association](metamodel-doc.md#association) or a [BinaryAssociation](metamodel-doc.md#binary-association) | |

### Example

<img src="/docs/img/associationClass.jpg" alt="Association Class" style="height: 60%; width:60%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Property, Multiplicity, BinaryAssociation, PrimitiveDataType, AssociationClass

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)

# Ends and BinaryAssociation definition
aend1: Property = Property(name="end1", owner=None, property_type=class1, multiplicity=Multiplicity(0, 1))
aend2: Property = Property(name="end2", owner=None, property_type=class2, multiplicity=Multiplicity(0, 1))
association: BinaryAssociation = BinaryAssociation(name="association1", ends={aend1, aend2})

# Attribute and AssociationClass definition
attribute1: Property = Property(name="attr", owner=None, property_type=PrimitiveDataType("int"))
association_class: AssociationClass = AssociationClass(name="AssociationCls", attributes={attribute1}, association=association)
```

## Generalization
> Generalization(general, specific)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- |------------| ----------- |-------|
| general | [Class](metamodel-doc.md#class) | General or parent Class | |
| specific | [Class](metamodel-doc.md#class) | Specific or child Class | |

### Example

<img src="/docs/img/generalization.jpg" alt="Generalization" style="height: 20%; width:20%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Generalization

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)

# Generalization definition
generalization: Generalization = Generalization(general=class1, specific=class2)
```

## GeneralizationSet
> GeneralizationSet(name, generalizations, is_disjoint, is_complete)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- | ---------- | ----------- | ------------- |
| name | str | Name of the GeneralizationSet | |
| generalizations | set([Generalization](metamodel-doc.md#generalization)) | Set of Generalizations | |
| is_disjoint | bool | Indicates whether or not the set of specific Classes in a Generalization relationship have instance in common. If *is_disjoint* is true, then the specific Classes of the GeneralizationSet have no members in common; that is, their intersection is empty  | |
| is_complete | bool | If *is_complete* is true, then every instance of the general Class is an instance of (at least) one of the specific Classes  | |

### Example

<img src="/docs/img/generalizationSet.jpg" alt="Generalization Set" style="height: 50%; width:50%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Generalization, GeneralizationSet

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)
class3: Class = Class(name="Cls3", attributes=None)

# Generalizations definition
generalization1: Generalization = Generalization(general=class1, specific=class2)
generalization2: Generalization = Generalization(general=class1, specific=class3)

# GeneralizationSet definition
generalization_set: GeneralizationSet = GeneralizationSet(name="GenSet", generalizations={generalization1, generalization2}, is_disjoint=True, is_complete=True)
```

## Package
> Package(name, classes)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- | ---------- | ----------- | ------------- |
| name | str | Name of the Package | |
| classes | set([Class](metamodel-doc.md#class)) | Set of Classes contained in the Package | |


### Example

<img src="/docs/img/package.jpg" alt="Package" style="height: 40%; width:40%;"/>

```python
# Import BUML classes
from metamodel.structural.structural import Class, Package

# Classes definition
class1: Class = Class(name="Cls1", attributes=None)
class2: Class = Class(name="Cls2", attributes=None)

# Package definition
package: Package = Package(name="Package", classes={class1, class2})
```

## Constraint
> Constraint(name, context, expression, language)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- | ---------- | ----------- | ------------- |
| name | str | Name of the Constraint | |
| context | [Class](metamodel-doc.md#class) | | |
| expression | Any |  | |
| language | str | | |

### Example
...

## DomainModel
> DomainModel(name, types, associations, generalizations, packages, constraints)

### Parameters

|   Name    |    Type    | Description | Default value |
| --------- | ---------- | ----------- | ------------- |
| name | str | Name of the DomainModel | |
| types | set[Type] | | |
| associationss | set[[Association](metamodel-doc.md#association)] |  | |
| generalizations | set[[Generalization](metamodel-doc.md#generalization)] | | |
| packages | set[[Package](metamodel-doc.md#package)] | | |
| constraints | set[[Constraint](metamodel-doc.md#constraint)] | | |

### Example

```python
# Import BUML classes
from metamodel.structural.structural import Class, Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Package, Constraint

# DomainModel definition, assuming that the model objects (class1, class2, assoc, etc.) have already been previously defined
model: DomainModel = DomainModel(name="model", types={class1,class2}, associations={assoc,binary_assoc}, generalizations={gen1,gen2}, packages={pkg}, constraints={const1, const2})
```