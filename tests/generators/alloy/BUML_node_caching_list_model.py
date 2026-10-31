import pytest

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, DomainModel, Multiplicity, IntegerType, Constraint,
)

# ---------------------------------------------------------------------------
# Fixture: node caching list domain model
# ---------------------------------------------------------------------------

@pytest.fixture
def node_caching_list_model():
    """The node caching linked list of Apache Commons Collections. Some OCL invariants may be redundant."""
    # Classes
    LinkedListNode = Class(name="LinkedListNode")
    NodeCachingLinkedList = Class(name="NodeCachingLinkedList")

    # LinkedListNode class attributes and methods
    LinkedListNode_value: Property = Property(name="value", type=IntegerType)
    LinkedListNode.attributes = {LinkedListNode_value}

    # NodeCachingLinkedList class attributes and methods
    NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE: Property = Property(
        name="DEFAULT_MAXIMUM_CACHE_SIZE", type=IntegerType
    )
    NodeCachingLinkedList_cacheSize: Property = Property(name="cacheSize", type=IntegerType)
    NodeCachingLinkedList_maximumCacheSize: Property = Property(name="maximumCacheSize", type=IntegerType)
    NodeCachingLinkedList_modCount: Property = Property(name="modCount", type=IntegerType)
    NodeCachingLinkedList_size: Property = Property(name="size", type=IntegerType)
    NodeCachingLinkedList.attributes = {
        NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE, NodeCachingLinkedList_cacheSize,
        NodeCachingLinkedList_maximumCacheSize, NodeCachingLinkedList_modCount, NodeCachingLinkedList_size,
    }

    # Relationships
    list_header: BinaryAssociation = BinaryAssociation(
        name="list_header",
        ends={
            Property(name="header_of", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
            Property(name="header", type=LinkedListNode, multiplicity=Multiplicity(1, 1))
        }
    )
    list_cache: BinaryAssociation = BinaryAssociation(
        name="list_cache",
        ends={
            Property(name="cache_of", type=NodeCachingLinkedList, multiplicity=Multiplicity(0, 1), is_navigable=False),
            Property(name="firstCachedNode", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
        }
    )
    node_next: BinaryAssociation = BinaryAssociation(
        name="node_next",
        ends={
            Property(name="next_of", type=LinkedListNode, multiplicity=Multiplicity(0, 1), is_navigable=False),
            Property(name="next", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
        }
    )
    node_previous: BinaryAssociation = BinaryAssociation(
        name="node_previous",
        ends={
            Property(name="previous_of", type=LinkedListNode, multiplicity=Multiplicity(0, 1), is_navigable=False),
            Property(name="previous", type=LinkedListNode, multiplicity=Multiplicity(0, 1))
        }
    )

    # OCL Constraints
    has_header: Constraint = Constraint(
        name="has_header",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv has_header: self.header <> null",
        language="OCL"
    )
    header_has_next_and_previous: Constraint = Constraint(
        name="header_has_next_and_previous",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv header_has_next_and_previous: "
                   "(self.header.next <> null) and (self.header.previous <> null)",
        language="OCL"
    )
    size_is_not_negative: Constraint = Constraint(
        name="size_is_not_negative",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv size_is_not_negative: self.size >= 0",
        language="OCL"
    )
    cache_within_maximum: Constraint = Constraint(
        name="cache_within_maximum",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv cache_within_maximum: self.cacheSize <= self.maximumCacheSize",
        language="OCL"
    )
    default_maximum_cache_size_is_6: Constraint = Constraint(
        name="default_maximum_cache_size_is_6",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv default_maximum_cache_size_is_6: "
                   "self.DEFAULT_MAXIMUM_CACHE_SIZE = 6",
        language="OCL"
    )
    size_counts_nodes_after_header: Constraint = Constraint(
        name="size_counts_nodes_after_header",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv size_counts_nodes_after_header: "
                   "self.size = self.header->closure(next)->size() - 1",
        language="OCL"
    )
    cache_size_counts_cached_nodes: Constraint = Constraint(
        name="cache_size_counts_cached_nodes",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv cache_size_counts_cached_nodes: "
                   "self.cacheSize = self.firstCachedNode->closure(next)->size()",
        language="OCL"
    )
    cached_nodes_unlinked_and_acyclic: Constraint = Constraint(
        name="cached_nodes_unlinked_and_acyclic",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv cached_nodes_unlinked_and_acyclic: "
                   "self.firstCachedNode->closure(next)->forAll(n: LinkedListNode | "
                   "(n.previous = null) and (n.next->closure(next)->excludes(n)))",
        language="OCL"
    )
    cached_nodes_have_no_value: Constraint = Constraint(
        name="cached_nodes_have_no_value",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv cached_nodes_have_no_value: "
                   "self.firstCachedNode->closure(next)->forAll(n: LinkedListNode | n.value = null)",
        language="OCL"
    )
    cached_nodes_are_acyclic: Constraint = Constraint(
        name="cached_nodes_are_acyclic",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv cached_nodes_are_acyclic: "
                   "self.firstCachedNode->closure(next)->forAll(n: LinkedListNode | "
                   "n.next->closure(next)->excludes(n))",
        language="OCL"
    )
    nodes_are_doubly_linked: Constraint = Constraint(
        name="nodes_are_doubly_linked",
        context=NodeCachingLinkedList,
        expression="context NodeCachingLinkedList inv nodes_are_doubly_linked: "
                   "self.header->closure(next)->forAll(n: LinkedListNode | "
                   "((n <> null and n.next <> null) and (n.previous <> null and n.next.previous = n)) "
                   "and (n.previous.next = n))",
        language="OCL"
    )

    # Domain Model
    model = DomainModel(
        name="NodeCachingLinkedList",
        types={LinkedListNode, NodeCachingLinkedList},
        associations={list_header, list_cache, node_next, node_previous},
        constraints={
            has_header, header_has_next_and_previous, size_is_not_negative, cache_within_maximum,
            default_maximum_cache_size_is_6, size_counts_nodes_after_header, cache_size_counts_cached_nodes,
            cached_nodes_unlinked_and_acyclic, cached_nodes_have_no_value, cached_nodes_are_acyclic,
            nodes_are_doubly_linked,
        },
    )
    return model
