# BUML PlantUML notation

You can also get a BUML model from a class diagram specified with [PlantUML](https://plantuml.com/). You will have to provide the class model using PlantUML textual notation and our transformation will produce the BUML based model. Let's see an example.

The following is a basic model written with PlantUML. Store this text in a file with `.buml` extension. For example, `hello_world.buml`

```
@startuml
   skinparam groupInheritance 2

   class A {
      -a1: int
      a2: bool
   }

   abstract class B {
      +b1: int
      #b2: float
      functionB()
   }

   class C {
      +c1: str
      +void functionC()
   }

   class D {
      +d1: str
      +void functionC()
   }

   A "1" o-- "1..*" D: AggregationAD
   A "1" *-- "1" C: CompositionAC
   D "1" -- "1..*" C: BidirectionalDC
   B <|-- A
   B <|-- C
@enduml
```

Then, load and process the model using our grammar and apply the transformation to obtain the BUML model.

```python
# Import textx and buml transformation
from textx import metamodel_from_file
from notations.textx.textx_to_buml import textx_to_buml

# Building metamodel from buml.tx grammar
buml_mm = metamodel_from_file('buml.tx')

# Transforming the textual model to plain Python objects
hello_world_buml_model = buml_mm.model_from_file('hello_world.buml')

# Transforming the Python objects to BUML model
domain: DomainModel = textx_to_buml(hello_world_buml_model)
```
> _Note: you must have two files: (1) the plantUML model `hello_world.buml`, and (2) the grammar `buml.tx` which can be found at this [link](https://github.com/BESSER-PEARL/BESSER-UML/blob/master/BUML/notations/textx/buml.tx)._


Now, you have the plantUML model transformed to the BUML metamodel in `domain`.