####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# Classes
LinkedListNode = Class(name="LinkedListNode")
NodeCachingLinkedList = Class(name="NodeCachingLinkedList")

# LinkedListNode class attributes and methods
LinkedListNode_value: Property = Property(name="value", type=IntegerType)
LinkedListNode.attributes={LinkedListNode_value}

# NodeCachingLinkedList class attributes and methods
NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE: Property = Property(name="DEFAULT_MAXIMUM_CACHE_SIZE", type=IntegerType)
NodeCachingLinkedList_cacheSize: Property = Property(name="cacheSize", type=IntegerType)
NodeCachingLinkedList_maximumCacheSize: Property = Property(name="maximumCacheSize", type=IntegerType)
NodeCachingLinkedList_modCount: Property = Property(name="modCount", type=IntegerType)
NodeCachingLinkedList_size: Property = Property(name="size", type=IntegerType)
NodeCachingLinkedList.attributes={NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE, NodeCachingLinkedList_cacheSize, NodeCachingLinkedList_maximumCacheSize, NodeCachingLinkedList_modCount, NodeCachingLinkedList_size}

# Relationships
NodeCachingLL_NodeLL: BinaryAssociation = BinaryAssociation(
    name="NodeCachingLL_NodeLL",
    ends={
        Property(name="rol_1", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="header", type=LinkedListNode, multiplicity=Multiplicity(1, 1))
    }
)
NodeCachingLL_NodeLL_1: BinaryAssociation = BinaryAssociation(
    name="NodeCachingLL_NodeLL_1",
    ends={
        Property(name="rol_2", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="firstCachedNode", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
    }
)
NodeLL_NodeLL: BinaryAssociation = BinaryAssociation(
    name="NodeLL_NodeLL",
    ends={
        Property(name="rol_4", type=LinkedListNode, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="next", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
    }
)
NodeLL_NodeLL_3: BinaryAssociation = BinaryAssociation(
    name="NodeLL_NodeLL_3",
    ends={
        Property(name="nodell_3", type=LinkedListNode, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="previous", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
    }
)


# OCL Constraints
NodeCachingLinkedList_inv_1_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_1_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_1_1 : self.header<>null",
    language="OCL"
)
NodeCachingLinkedList_inv_3_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_3_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_3_1 : (self.size) >=0",
    language="OCL"
)
NodeCachingLinkedList_inv_2_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_2_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_2_1 : (self.header.next<>null) and (self.header.previous<>null)",
    language="OCL"
)
NodeCachingLinkedList_inv_4_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_4_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_4_1 :  self.cacheSize <= self.maximumCacheSize",
    language="OCL"
)
NodeCachingLinkedList_inv_5_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_5_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_5_1 : self.DEFAULT_MAXIMUM_CACHE_SIZE=6",
    language="OCL"
)
NodeCachingLinkedList_inv_6_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_6_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_6_1 : (self.size) = self.header->closure(next)->excluding(null) -> size()-1",
    language="OCL"
)
NodeCachingLinkedList_inv_7_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_7_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_7_1 : self.cacheSize = self.firstCachedNode->closure(next)->excluding(null) -> size()",
    language="OCL"
)
NodeCachingLinkedList_inv_8_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_8_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_8_1 : self.firstCachedNode->closure(next)->excluding(null)->forAll(n:LinkedListNode| (n.previous=null) and  (n.next->closure(next)->excludes(n)))",
    language="OCL"
)
NodeCachingLinkedList_inv_9_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_9_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_9_1 : self.firstCachedNode->closure(next)->excluding(null)->forAll(n:LinkedListNode|  n.value=null)",
    language="OCL"
)
NodeCachingLinkedList_inv_10_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_10_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_10_1 : self.firstCachedNode->closure(next)->excluding(null)->forAll(n:LinkedListNode| (n.next->closure(next)->excludes(n)))",
    language="OCL"
)
NodeCachingLinkedList_inv_11_1: Constraint = Constraint(
    name="NodeCachingLinkedList_inv_11_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv NodeCachingLinkedList_inv_11_1 : self.header->closure(next)->excluding(null)->forAll(n:LinkedListNode |((n<>null and n.next<>null) and (n.previous<>null and n.next.previous=n)) and (n.previous.next=n))",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={LinkedListNode, NodeCachingLinkedList},
    associations={NodeCachingLL_NodeLL, NodeCachingLL_NodeLL_1, NodeLL_NodeLL, NodeLL_NodeLL_3},
    constraints={NodeCachingLinkedList_inv_1_1, NodeCachingLinkedList_inv_3_1, NodeCachingLinkedList_inv_2_1, NodeCachingLinkedList_inv_4_1, NodeCachingLinkedList_inv_5_1, NodeCachingLinkedList_inv_6_1, NodeCachingLinkedList_inv_7_1, NodeCachingLinkedList_inv_8_1, NodeCachingLinkedList_inv_9_1, NodeCachingLinkedList_inv_10_1, NodeCachingLinkedList_inv_11_1},
    generalizations={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="The OCL invariant may be redundant.")
project = Project(
    name="prueba_de_concepto_NCLL",
    models=[domain_model],
    owner="nnn",
    metadata=metadata
)
