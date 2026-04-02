"""Tests for the LLM model serializer."""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Metadata,
    Method,
    Multiplicity,
    Parameter,
    PrimitiveDataType,
    Property,
    UNLIMITED_MAX_MULTIPLICITY,
    AssociationClass,
)
from besser.generators.llm.model_serializer import serialize_domain_model


class TestSerializeDomainModel:

    def _build_blog_model(self) -> DomainModel:
        """Build a realistic blog model for testing."""
        StringType = PrimitiveDataType("str")
        IntegerType = PrimitiveDataType("int")
        BooleanType = PrimitiveDataType("bool")

        status = Enumeration(name="PostStatus", literals={
            EnumerationLiteral(name="DRAFT"),
            EnumerationLiteral(name="PUBLISHED"),
            EnumerationLiteral(name="ARCHIVED"),
        })

        user = Class(name="User", metadata=Metadata(description="A registered user"))
        user.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="email", type=StringType),
            Property(name="username", type=StringType),
            Property(name="bio", type=StringType, is_optional=True),
        }

        post = Class(name="Post")
        post.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="title", type=StringType),
            Property(name="content", type=StringType),
            Property(name="published", type=BooleanType),
        }
        post.methods = {
            Method(name="word_count", type=IntegerType),
        }

        comment = Class(name="Comment")
        comment.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="text", type=StringType),
        }

        user_post = BinaryAssociation(name="User_Post", ends={
            Property(name="author", type=user, multiplicity=Multiplicity(1, 1)),
            Property(name="posts", type=post, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        })
        post_comment = BinaryAssociation(name="Post_Comment", ends={
            Property(name="post", type=post, multiplicity=Multiplicity(1, 1)),
            Property(name="comments", type=comment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        })

        gen = Generalization(general=user, specific=post)

        constraint = Constraint(
            name="valid_email",
            context=user,
            expression="self.email->matches('[a-zA-Z0-9.]+@[a-zA-Z0-9.]+')",
            language="OCL",
        )

        return DomainModel(
            name="BlogApp",
            types={user, post, comment, status},
            associations={user_post, post_comment},
            generalizations={gen},
            constraints={constraint},
        )

    def test_basic_structure(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert result["name"] == "BlogApp"
        assert "classes" in result
        assert "enumerations" in result
        assert "associations" in result
        assert "generalizations" in result
        assert "constraints" in result

    def test_classes_sorted_by_name(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        names = [c["name"] for c in result["classes"]]
        assert names == sorted(names)

    def test_class_attributes(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        user = next(c for c in result["classes"] if c["name"] == "User")
        attr_names = {a["name"] for a in user["attributes"]}
        assert "email" in attr_names
        assert "username" in attr_names
        assert "bio" in attr_names
        assert "id" in attr_names

        # Check id is marked
        id_attr = next(a for a in user["attributes"] if a["name"] == "id")
        assert id_attr.get("is_id") is True

        # Check optional
        bio_attr = next(a for a in user["attributes"] if a["name"] == "bio")
        assert bio_attr.get("is_optional") is True

    def test_methods_serialized(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        post = next(c for c in result["classes"] if c["name"] == "Post")
        assert "methods" in post
        wc = next(m for m in post["methods"] if m["name"] == "word_count")
        assert wc["return_type"] == "int"

    def test_enumerations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["enumerations"]) == 1
        enum = result["enumerations"][0]
        assert enum["name"] == "PostStatus"
        assert set(enum["literals"]) == {"DRAFT", "PUBLISHED", "ARCHIVED"}

    def test_associations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assoc_names = {a["name"] for a in result["associations"]}
        assert "User_Post" in assoc_names
        assert "Post_Comment" in assoc_names

        user_post = next(a for a in result["associations"] if a["name"] == "User_Post")
        assert len(user_post["ends"]) == 2
        author_end = next(e for e in user_post["ends"] if e["role"] == "author")
        assert author_end["multiplicity"] == "1..1"
        posts_end = next(e for e in user_post["ends"] if e["role"] == "posts")
        assert posts_end["multiplicity"] == "0..*"

    def test_generalizations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["generalizations"]) == 1
        assert result["generalizations"][0]["parent"] == "User"
        assert result["generalizations"][0]["child"] == "Post"

    def test_constraints(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["constraints"]) == 1
        c = result["constraints"][0]
        assert c["context"] == "User"
        assert "email" in c["expression"]

    def test_metadata_included(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        user = next(c for c in result["classes"] if c["name"] == "User")
        assert user["metadata"]["description"] == "A registered user"

    def test_empty_model(self):
        model = DomainModel(name="Empty")
        result = serialize_domain_model(model)
        assert result["name"] == "Empty"
        assert "classes" not in result
        assert "associations" not in result

    def test_compact_output(self):
        """Verify only non-empty sections are included (saves tokens)."""
        StringType = PrimitiveDataType("str")
        cls = Class(name="Simple")
        cls.attributes = {Property(name="name", type=StringType)}
        model = DomainModel(name="Minimal", types={cls})
        result = serialize_domain_model(model)

        assert "associations" not in result
        assert "generalizations" not in result
        assert "constraints" not in result
        assert "enumerations" not in result
