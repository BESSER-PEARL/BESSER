#!/usr/bin/env python3

from besser.BUML.metamodel.structural import (
    StringType, IntegerType, FloatType, BooleanType,
    Property
)

# Let's debug what these type objects look like
print("IntegerType:", IntegerType)
print("IntegerType type:", type(IntegerType))
print("IntegerType __name__:", getattr(IntegerType, '__name__', 'No __name__'))

# Create a property with IntegerType
test_prop = Property(name="test", type=IntegerType)
print("\nProperty type:", test_prop.type)
print("Property type type:", type(test_prop.type))
print("Property type __name__:", getattr(test_prop.type, '__name__', 'No __name__'))
print("Property type == IntegerType:", test_prop.type == IntegerType)
print("Property type is IntegerType:", test_prop.type is IntegerType)

# Check if it has __class__
if hasattr(test_prop.type, '__class__'):
    print("Property type __class__:", test_prop.type.__class__)
    print("Property type __class__.__name__:", test_prop.type.__class__.__name__)
