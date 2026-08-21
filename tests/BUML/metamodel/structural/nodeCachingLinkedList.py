####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)

# Classes
NodeCachingLinkedList = Class(name="NodeCachingLinkedList")
LinkedListNode = Class(name="LinkedListNode")
Integer = Class(name="Integer")


# NodeCachingLinkedList class attributes and methods
NodeCachingLinkedList_listSize: Property = Property(name="size", type=IntegerType)
NodeCachingLinkedList_cacheSize: Property = Property(name="cacheSize", type=IntegerType)
NodeCachingLinkedList_modCount: Property = Property(name="modCount", type=IntegerType)

NodeCachingLinkedList_maximumCacheSize: Property = Property(name="maximumCacheSize", type=IntegerType)

NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE: Property = Property(name="DEFAULT_MAXIMUM_CACHE_SIZE", type=IntegerType)
NodeCachingLinkedList.attributes={NodeCachingLinkedList_maximumCacheSize,NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE, NodeCachingLinkedList_cacheSize, NodeCachingLinkedList_listSize, NodeCachingLinkedList_modCount}

# LinkedListNode class attributes and methods
    #LinkedListNode_value: Property = Property(name="value", type=IntegerType)
    #LinkedListNode.attributes={LinkedListNode_value}

#LinkedListNode_Object: BinaryAssociation = BinaryAssociation(
#    name="LinkedListNode_Object",
#    ends={
#        Property(name="linkedlistnode", type=LinkedListNode, multiplicity=Multiplicity(0, 9999), is_navigable=False),
#        Property(name="value", type=Object, multiplicity=Multiplicity(0, 1))
#    }
#)







# Relationships

LL_1: BinaryAssociation = BinaryAssociation(
    name="LL_1",
    ends={
        Property(name="rol_3", type=LinkedListNode, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="value", type=Integer, multiplicity=Multiplicity(0, 1))
    }
)




NodeCachingLL_NodeLL_1: BinaryAssociation = BinaryAssociation(
    name="NodeCachingLL_NodeLL_1",
    ends={
        Property(name="rol_2", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="firstCachedNode", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
    }
)


NodeCachingLL_NodeLL: BinaryAssociation = BinaryAssociation(
    name="NodeCachingLL_NodeLL",
    ends={
        Property(name="rol_1", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property(name="header", type=LinkedListNode, multiplicity=Multiplicity(1, 1))
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
#    @Invariant 
#		( this.header!=null ) &&
inv_1: Constraint = Constraint(
    name="inv_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.header<>null",
    language="OCL"
)

inv_1_1: Constraint = Constraint(
    name="inv_1_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.header.value=null",
    language="OCL"
)

# 		( this.header.next!=null ) &&
#		( this.header.previous!=null ) &&

inv_2: Constraint = Constraint(
    name="inv_2",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : (self.header.next<>null) and (self.header.previous<>null))",
    language="OCL"
)

# 		( this.size>=0 ) &&

inv_3: Constraint = Constraint(
    name="inv_3",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : (self.size) >=0",
    language="OCL"
)

# 		( this.cacheSize <= this.maximumCacheSize ) &&

inv_4: Constraint = Constraint(
    name="inv_4",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv :  self.cacheSize <= self.maximumCacheSize",
    language="OCL"
)

# 		( this.DEFAULT_MAXIMUM_CACHE_SIZE == 6 ) &&

inv_5: Constraint = Constraint(
    name="inv_5",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.DEFAULT_MAXIMUM_CACHE_SIZE=6",
    language="OCL"
)


# 		( this.size==#(this.header.*next @- null)-1 ) &&


inv_6: Constraint = Constraint(
    name="inv_6",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : (self.size) = self.header->closure(next)->excluding(null) -> size()-1",
    language="OCL"
)

# 		( this.cacheSize == #(this.firstCachedNode.*next @- null) ) &&

inv_7: Constraint = Constraint(
    name="inv_7",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.cacheSize = self.firstCachedNode->closure(next)->excluding(null) -> size()",
    language="OCL"
)

# 		(all m: LinkedListNode | ( m in this.firstCachedNode.*next @- null ) => (
# 				  m !in m.next.*next @- null &&
# 				  m.previous==null &&
# 				  m.value==null )) &&

inv_8: Constraint = Constraint(
    name="inv_8",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.firstCachedNode->closure(next)->excluding(null)->" \
    "forAll(n:LinkedListNode| (n.previous=null) and  (n.next->closure(next)->excludes(n)))",
    language="OCL"
)



inv_83: Constraint = Constraint(
    name="inv_83",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.firstCachedNode->closure(next)->excluding(null)->forAll(n:LinkedListNode|  n.value=null)",
    language="OCL"
)




#Este invariante no obliga a que los nodos de la lista cache tengan el next y previous null, 
inv_8_1: Constraint = Constraint(
    name="inv_8_1",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.firstCachedNode->closure(next)->excluding(null)->" \
    "forAll(n:LinkedListNode| (n.next->closure(next)->excludes(n)))",
    language="OCL"
)





# 		(all n: LinkedListNode | ( n in this.header.*next @- null ) => (
# 				  n!=null &&
# 				  n.previous!=null &&
# 				  n.previous.next==n &&
# 				  n.next!=null &&
# 				  n.next.previous==n )) ; 


inv_9: Constraint = Constraint(
    name="inv_9",
    context=NodeCachingLinkedList,
    expression="context NodeCachingLinkedList inv : self.header->closure(next)" \
    "->excluding(null)->forAll(n:LinkedListNode |((n<>null and n.next<>null) and (n.previous<>null and n.next.previous=n)) and (n.previous.next=n))",
    language="OCL"
)






# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={NodeCachingLinkedList, LinkedListNode,Integer},
    associations={LL_1,NodeLL_NodeLL_3, NodeLL_NodeLL, NodeCachingLL_NodeLL, NodeCachingLL_NodeLL_1},
    #constraints={inv_1, inv_2, inv_3, inv_4, inv_5, inv_6, inv_7, inv_8, inv_9},
    constraints={inv_1, inv_1_1, inv_2, inv_3, inv_4,inv_5,inv_6,inv_7,inv_8,inv_9,inv_83},
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

import os
from besser.generators.alloy_generator import AlloyGenerator


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
alloy_model = AlloyGenerator(model=domain_model, output_dir=OUTPUT_DIR)
alloy_model.generate()