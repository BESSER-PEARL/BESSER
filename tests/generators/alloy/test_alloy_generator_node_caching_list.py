import pytest
from tests.generators.alloy.BUML_node_caching_list_model import node_caching_list_model  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def spec(node_caching_list_model):  # noqa: F811
    return translate(node_caching_list_model, scope=5)


def test_library_modules(spec):
    # No String, Date or Boolean types: no library module is opened
    assert len(spec.opens) == 0


def test_signatures(spec):
    assert len(spec.sigs) == 2
    assert "sig NodeCachingLinkedList" in spec.sigs
    assert "sig LinkedListNode" in spec.sigs


def test_signature_fields(spec):
    assert len(spec.fields["NodeCachingLinkedList"]) == 7
    assert "NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE: Int" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_cacheSize: Int" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_maximumCacheSize: Int" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_modCount: Int" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_size: Int" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_header: one LinkedListNode" in spec.fields["NodeCachingLinkedList"]
    assert "NodeCachingLinkedList_firstCachedNode: lone LinkedListNode" in spec.fields["NodeCachingLinkedList"]

    # The associations are reflexive over LinkedListNode: only the navigable ends (next, previous) become fields
    assert len(spec.fields["LinkedListNode"]) == 3
    assert "LinkedListNode_value: Int" in spec.fields["LinkedListNode"]
    assert "LinkedListNode_next: lone LinkedListNode" in spec.fields["LinkedListNode"]
    assert "LinkedListNode_previous: lone LinkedListNode" in spec.fields["LinkedListNode"]


def test_facts(spec):
    assert len(spec.facts) == 18

    # LinkedListNode -> LinkedListNode, next and previous (0..1)
    assert "all a: LinkedListNode | #(a.LinkedListNode_next) <= 1" in spec.facts
    assert "all a: LinkedListNode | #(a.LinkedListNode_previous) <= 1" in spec.facts
    # NodeCachingLinkedList -> LinkedListNode, first cached node (0..1)
    assert "all a: NodeCachingLinkedList | #(a.NodeCachingLinkedList_firstCachedNode) <= 1" in spec.facts
    # next_of, previous_of, header_of and cache_of (0..1) are not navigable, so there is no `lone` field
    # enforcing them: their bounds are checked navigating the opposite field backwards
    assert "all a: LinkedListNode | #(LinkedListNode_next.a) <= 1" in spec.facts
    assert "all a: LinkedListNode | #(LinkedListNode_previous.a) <= 1" in spec.facts
    assert "all a: LinkedListNode | #(NodeCachingLinkedList_header.a) <= 1" in spec.facts
    assert "all a: LinkedListNode | #(NodeCachingLinkedList_firstCachedNode.a) <= 1" in spec.facts

    # OCL: the list has a header (self.header <> null)
    assert "all self:this/NodeCachingLinkedList|(some self.NodeCachingLinkedList_header)" in spec.facts
    # OCL: the header has a next and a previous node
    assert "all self:this/NodeCachingLinkedList|((some self.NodeCachingLinkedList_header.LinkedListNode_next) " \
        "and (some self.NodeCachingLinkedList_header.LinkedListNode_previous))" in spec.facts
    # OCL: self.size >= 0
    assert "all self:this/NodeCachingLinkedList|(self.NodeCachingLinkedList_size >= 0)" in spec.facts
    # OCL: self.cacheSize <= self.maximumCacheSize
    assert "all self:this/NodeCachingLinkedList|" \
        "(self.NodeCachingLinkedList_cacheSize <= self.NodeCachingLinkedList_maximumCacheSize)" in spec.facts
    # OCL: self.DEFAULT_MAXIMUM_CACHE_SIZE = 6
    assert "all self:this/NodeCachingLinkedList|(self.NodeCachingLinkedList_DEFAULT_MAXIMUM_CACHE_SIZE = 6)" \
        in spec.facts
    # OCL: the size counts the nodes after the header (self.header->closure(next)->size() - 1)
    assert "all self:this/NodeCachingLinkedList|(self.NodeCachingLinkedList_size = " \
        "minus[#(((self.NodeCachingLinkedList_header).*(LinkedListNode_next))), 1])" in spec.facts
    # OCL: the cache size counts the cached nodes (self.firstCachedNode->closure(next)->size())
    assert "all self:this/NodeCachingLinkedList|(self.NodeCachingLinkedList_cacheSize = " \
        "#(((self.NodeCachingLinkedList_firstCachedNode).*(LinkedListNode_next))))" in spec.facts
    # OCL: the cached nodes have no previous node and are acyclic (forAll over a closure)
    assert "all self:this/NodeCachingLinkedList|all n : " \
        "((self.NodeCachingLinkedList_firstCachedNode).*(LinkedListNode_next)) | " \
        "(( no(n.LinkedListNode_previous)) and (n not in ((n.LinkedListNode_next).*(LinkedListNode_next))))" \
        in spec.facts
    # OCL: the cached nodes have no value (n.value = null)
    assert "all self:this/NodeCachingLinkedList|all n : " \
        "((self.NodeCachingLinkedList_firstCachedNode).*(LinkedListNode_next)) | ( no(n.LinkedListNode_value))" \
        in spec.facts
    # OCL: the cached nodes are acyclic
    assert "all self:this/NodeCachingLinkedList|all n : " \
        "((self.NodeCachingLinkedList_firstCachedNode).*(LinkedListNode_next)) | " \
        "(n not in ((n.LinkedListNode_next).*(LinkedListNode_next)))" in spec.facts
    # OCL: the nodes from the header are doubly linked
    assert "all self:this/NodeCachingLinkedList|all n : " \
        "((self.NodeCachingLinkedList_header).*(LinkedListNode_next)) | " \
        "((((some n) and (some n.LinkedListNode_next)) and ((some n.LinkedListNode_previous) and " \
        "(n.LinkedListNode_next.LinkedListNode_previous = n))) and " \
        "(n.LinkedListNode_previous.LinkedListNode_next = n))" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 NodeCachingLinkedList" in spec.run_scopes
    assert "5 LinkedListNode" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
