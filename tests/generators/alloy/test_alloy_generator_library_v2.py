import os
import pytest
from tests.generators.alloy.BUML_library_with_ocl_v2 import library_with_ocl_v2
from besser.generators.alloy.alloy_generator_new import AlloyGeneratorNew
from tests.generators.alloy.utils_alloy_translation import AlloyTestSpec


@pytest.fixture
def spec(library_with_ocl_v2, tmpdir):
    tmpdir = tmpdir.mkdir("output")
    generator = AlloyGeneratorNew(model=library_with_ocl_v2, output_dir=str(tmpdir))
    generator.generate()
    als_file = generator.get_specification_path()
    assert os.path.isfile(als_file), f"Alloy specification not generated at {als_file}"
    # Date and String attributes require the date and strings libraries; `false` requires booleans
    assert sorted(os.listdir(str(tmpdir))) == ["boolean.als", "date.als", "model.als", "strings.als"]
    with open(als_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "open boolean" in content
    assert "open date" in content
    assert "open strings" in content
    assert "open util/integer" not in content
    assert "open util/ordering" not in content
    return AlloyTestSpec(content)


def test_signatures(spec):
    assert "sig Library" in spec.sigs
    assert "sig Book" in spec.sigs
    assert "sig Author" in spec.sigs
    # 5 strings: the OCL string literals, plus others filling the scope up
    assert len(spec.subsigs_of("Str")) == 5
    # OCL string literals: one constant each, named after its value
    assert "one sig Str_cortazar extends Str" in spec.sigs  # Author_inv_10_1
    assert "one sig Str_borges extends Str" in spec.sigs  # Author_inv_9_1
    # The other strings fill the scope up
    assert "one sig Str_str0 extends Str" in spec.sigs
    assert "one sig Str_str1 extends Str" in spec.sigs
    assert "one sig Str_str2 extends Str" in spec.sigs
    # 5 dates, all random (seed 0) as there are no OCL date literals, sorted chronologically
    assert len(spec.subsigs_of("Date")) == 5
    assert "one sig Date_2001_10_25 extends Date" in spec.sigs
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2018_11_12 extends Date" in spec.sigs
    assert "one sig Date_2033_12_31 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs


def test_signature_fields(spec):
    assert len(spec.fields["Library"]) == 5
    assert "Library_name: Str" in spec.fields["Library"]
    assert "Library_web_page: Str" in spec.fields["Library"]
    assert "Library_address: Str" in spec.fields["Library"]
    assert "Library_telephone: Str" in spec.fields["Library"]
    assert "Library_books: set Book" in spec.fields["Library"]

    assert len(spec.fields["Book"]) == 8
    assert "Book_title: Str" in spec.fields["Book"]
    assert "Book_pages: Int" in spec.fields["Book"]
    assert "Book_stock: Int" in spec.fields["Book"]
    assert "Book_price: Int" in spec.fields["Book"]  # FloatType -> Int
    assert "Book_release: Date" in spec.fields["Book"]
    assert "Book_genre: Genre" in spec.fields["Book"]
    assert "Book_library: set Library" in spec.fields["Book"]
    assert "Book_authors: set Author" in spec.fields["Book"]

    assert len(spec.fields["Author"]) == 3
    assert "Author_name: Str" in spec.fields["Author"]
    assert "Author_birth: Date" in spec.fields["Author"]
    assert "Author_books: set Book" in spec.fields["Author"]

    assert len(spec.fields["Str_cortazar"]) == 0
    assert len(spec.fields["Str_borges"]) == 0


def test_enumerations(spec):
    assert len(spec.subsigs_of("Genre")) == 10
    assert "abstract sig Genre" in spec.sigs
    assert "one sig ENUM_Genre_Poetry extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Thriller extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_History extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Technology extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Romance extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Horror extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Adventure extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Philosophy extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Cookbooks extends Genre" in spec.sigs
    assert "one sig ENUM_Genre_Fantasy extends Genre" in spec.sigs


def test_facts(spec):
    assert len(spec.facts) == 11

    # Library <-> Book: a book is in at least one library (1..*)
    assert "all a: Book | #(a.Book_library) >= 1" in spec.facts
    assert "Library_books = ~Book_library" in spec.facts
    # Author <-> Book: a book has at least one author (1..*)
    assert "all a: Book | #(a.Book_authors) >= 1" in spec.facts
    assert "Book_authors = ~Author_books" in spec.facts
    # Dates: their chronological order in the Date ordering
    assert "dateLt[Date_2001_10_25, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2018_11_12] " \
        "and dateLt[Date_2018_11_12, Date_2033_12_31] and dateLt[Date_2033_12_31, Date_2037_11_17]" in spec.facts

    # String literals: uninterpreted constants have no characters
    # assert "Str_cortazar.Str_data = 0->c + 1->o + 2->r + 3->t + 4->a + 5->z + 6->a + 7->r" in spec.facts
    # assert "Str_borges.Str_data = 0->b + 1->o + 2->r + 3->g + 4->e + 5->s" in spec.facts

    # OCL: book_positive_pages (self.pages > 0)
    assert "all self:this/Book|(self.Book_pages > 0)" in spec.facts
    # OCL: book_nonneg_stock (self.stock >= 0)
    assert "all self:this/Book|(self.Book_stock >= 0)" in spec.facts
    # OCL: book_has_title (String size() -> len), disabled: uninterpreted strings have no length
    # assert "all self:this/Book|(len[self.Book_title] > 0)" in spec.facts
    # OCL: book_nonneg_price (self.price >= 0)
    assert "all self:this/Book|(self.Book_price >= 0)" in spec.facts
    # OCL: library_has_books (collection size() -> cardinality)
    assert "all self:this/Library|(#(self.Library_books) > 0)" in spec.facts
    # OCL: library_named (String size() -> len), disabled: uninterpreted strings have no length
    # assert "all self:this/Library|(len[self.Library_name] > 0)" in spec.facts
    # OCL: author_named (String size() -> len), disabled: uninterpreted strings have no length
    # assert "all self:this/Author|(len[self.Author_name] > 0)" in spec.facts
    # OCL: Author_inv_8_1, books released after the author's birth (forAll + date comparison),
    # disabled with its invariant: date ordering is unsupported
    # assert "all self:this/Author|all b : self.Author_books | " \
    #     "(dateGte[b.Book_release,self.Author_birth])" in spec.facts
    # OCL: Author_inv_9_1, 'borges' (Str_borges) has written at least one book
    assert "all self:this/Author|((strEq[self.Author_name,Str_borges]) => " \
        "(#(self.Author_books) > 0))" in spec.facts
    # OCL: Author_inv_10_1, if 'cortazar' (Str_cortazar) then more than 2 books over 10 pages else false
    assert "all self:this/Author|((strEq[self.Author_name,Str_cortazar]) => " \
        "(#({select0[self.Author_books]}) > 2) else (False = True))" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Library" in spec.run_scopes
    assert "5 Book" in spec.run_scopes
    assert "5 Author" in spec.run_scopes
    assert "5 Date" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
