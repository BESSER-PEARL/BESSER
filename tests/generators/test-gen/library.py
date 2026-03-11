from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation
)
from besser.generators.python_classes.python_classes_generator import PythonGenerator
from besser.generators.testgen.test_generator import TestGenerator  # [5]

# =============================================================================
# 1. Define Enumerations
# =============================================================================
genre_lit_fiction    = EnumerationLiteral(name="FICTION")
genre_lit_nonfiction = EnumerationLiteral(name="NONFICTION")
genre_lit_science    = EnumerationLiteral(name="SCIENCE")

genre_enum = Enumeration(
    name="Genre",
    literals={genre_lit_fiction, genre_lit_nonfiction, genre_lit_science}
)

# =============================================================================
# 2. Define Attributes
# =============================================================================
title_prop  = Property(name="title",  type=StringType,  multiplicity=Multiplicity(1, 1))
pages_prop  = Property(name="pages",  type=IntegerType, multiplicity=Multiplicity(1, 1))
rating_prop = Property(name="rating", type=FloatType,   multiplicity=Multiplicity(1, 1))
inprint_prop = Property(name="inPrint", type=BooleanType, multiplicity=Multiplicity(1, 1))

name_prop   = Property(name="name",   type=StringType,  multiplicity=Multiplicity(1, 1))
email_prop  = Property(name="email",  type=StringType,  multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Methods
# =============================================================================
get_summary = Method(
    name="getSummary",
    parameters=[Parameter(name="maxLength", type=IntegerType)],
)

is_available = Method(
    name="isAvailable",
    parameters=[],
)

# =============================================================================
# 4. Define Classes
# =============================================================================
book = Class(
    name="Book",
    attributes={title_prop, pages_prop, rating_prop, inprint_prop},
    methods={get_summary, is_available},
)

author = Class(
    name="Author",
    attributes={name_prop, email_prop},
)

# =============================================================================
# 5. Define Association: Author --< Book (one author, many books)
# =============================================================================
author_end = Property(
    name="author",
    type=author,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True,
)
books_end = Property(
    name="books",
    type=book,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True,
)
wrote_assoc = BinaryAssociation(
    name="Wrote",
    ends={author_end, books_end},
)

# =============================================================================
# 6. Build the DomainModel — pass all types at construction time
# =============================================================================
model = DomainModel(
    name="LibraryModel",
    types={book, author, genre_enum},
    associations={wrote_assoc},
)

# =============================================================================
# 7. Sanity check
# =============================================================================
print("Classes in model:")
for c in model.classes_sorted_by_inheritance():
    print(f"  {'[abstract] ' if c.is_abstract else ''}{c.name}")
    print(f"    attributes : {[a.name for a in c.attributes]}")
    print(f"    methods    : {[m.name for m in c.methods]}")

print("\nEnumerations:")
for e in model.get_enumerations():
    print(f"  {e.name} → {[l.name for l in e.literals]}")

# =============================================================================
# 8. Step 1 — Generate Python domain classes first (required import in tests)
# =============================================================================
python_gen = PythonGenerator(model=model, output_dir="output")
python_gen.generate()
# Produces: output/classes.py

# =============================================================================
# 9. Step 2 — Generate Hypothesis + pytest test suite [5]
# =============================================================================
test_gen = TestGenerator(model=model, output_dir="output")
test_gen.generate()
# Produces: output/test_hypothesis.py


