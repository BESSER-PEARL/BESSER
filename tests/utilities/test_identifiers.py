"""The three identifier sanitizers, pinned.

``safe_var_name``, ``owl2uml.naming.sanitize`` and ``_common.sanitize_python_identifier``
all run on ``besser.utilities.identifiers.sanitize_identifier``. Their outputs are what
saved projects, generated code and the golden outputs depend on, so the table below was
recorded from the implementations *before* they shared a core, and must not change with it.

The one deliberate change since: every output is now a valid Python identifier. A name
that starts with a digit used to come back from ``safe_var_name`` still starting with one
("9lives"), because the tidy-up stripped the ``_`` prefix again, and characters such as
``²`` passed ``str.isalnum()`` but not the parser. Those rows now get the prefix, and
``²`` is NFKC-normalised to ``2`` (as the parser itself would read it).
"""

import pytest

from besser.BUML.notations.kg_to_buml._common import sanitize_python_identifier
from besser.BUML.notations.kg_to_buml.owl2uml.naming import sanitize
from besser.utilities.buml_code_builder.common import safe_var_name
from besser.utilities.identifiers import sanitize_identifier

# (input, safe_var_name, safe_var_name(lowercase=False), naming.sanitize,
#  sanitize_python_identifier(.., "n"), sanitize_python_identifier(.., "node"))
ROWS = [
    ('', 'unnamed', 'unnamed', '_', 'n', 'node'),
    (' ', 'unnamed', 'unnamed', '_', 'n', 'node'),
    ('a b', 'a_b', 'a_b', 'a_b', 'a_b', 'a_b'),
    ('a-b', 'a_b', 'a_b', 'a_b', 'a_b', 'a_b'),
    ('9lives', '_9lives', '_9lives', '_9lives', 'n_9lives', 'node_9lives'),
    ('_9', '_9', '_9', '_9', 'n_9', 'node_9'),
    ('__a__b__', 'a_b', 'a_b', '__a__b__', 'a__b', 'a__b'),
    ('Café', 'café', 'Café', 'Caf_', 'Caf', 'Caf'),
    ('naïve façade', 'naïve_façade', 'naïve_façade', 'na_ve_fa_ade', 'na_ve_fa_ade', 'na_ve_fa_ade'),
    ('class', 'class_', 'class_', 'class', 'class_', 'class_'),
    ('Class', 'class_', 'Class', 'Class', 'Class', 'Class'),
    ('from', 'from_', 'from_', 'from', 'from_', 'from_'),
    ('_', 'unnamed', 'unnamed', '_', 'n', 'node'),
    ('___', 'unnamed', 'unnamed', '___', 'n', 'node'),
    ('-', 'unnamed', 'unnamed', '_', 'n', 'node'),
    ('a.b/c#d', 'a_b_c_d', 'a_b_c_d', 'a_b_c_d', 'a_b_c_d', 'a_b_c_d'),
    ('日本語', '日本語', '日本語', '___', 'n', 'node'),
    ('x²', 'x2', 'x2', 'x_', 'x', 'x'),
    ('٣abc', '_٣abc', '_٣abc', '_abc', 'abc', 'abc'),
    ('a\nb', 'a_b', 'a_b', 'a_b', 'a_b', 'a_b'),
    ('tab\t', 'tab', 'tab', 'tab_', 'tab', 'tab'),
    ('  padded  ', 'padded', 'padded', '__padded__', 'padded', 'padded'),
    ('UPPER lower', 'upper_lower', 'UPPER_lower', 'UPPER_lower', 'UPPER_lower', 'UPPER_lower'),
    ('a__b', 'a_b', 'a_b', 'a__b', 'a__b', 'a__b'),
    ('_leading', 'leading', 'leading', '_leading', 'leading', 'leading'),
    ('trailing_', 'trailing', 'trailing', 'trailing_', 'trailing', 'trailing'),
    ('123', '_123', '_123', '_123', 'n_123', 'node_123'),
    ('1_2', '_1_2', '_1_2', '_1_2', 'n_1_2', 'node_1_2'),
    ('with😀emoji', 'with_emoji', 'with_emoji', 'with_emoji', 'with_emoji', 'with_emoji'),
    ('None', 'none', 'None_', 'None', 'None_', 'None_'),
    ('async', 'async_', 'async_', 'async', 'async_', 'async_'),
    ('match', 'match', 'match', 'match', 'match', 'match'),
    ('aaa', 'aaa', 'aaa', 'aaa', 'aaa', 'aaa'),
    ('Person', 'person', 'Person', 'Person', 'Person', 'Person'),
    ('has-part', 'has_part', 'has_part', 'has_part', 'has_part', 'has_part'),
    ('http://ex.org/Thing', 'http_ex_org_thing', 'http_ex_org_Thing', 'http___ex_org_Thing', 'http___ex_org_Thing', 'http___ex_org_Thing'),
    ('my class', 'my_class', 'my_class', 'my_class', 'my_class', 'my_class'),
    ('_some_p_D', 'some_p_d', 'some_p_D', '_some_p_D', 'some_p_D', 'some_p_D'),
    ('_C1_C2_Union', 'c1_c2_union', 'C1_C2_Union', '_C1_C2_Union', 'C1_C2_Union', 'C1_C2_Union'),
    ('ß', 'ß', 'ß', '_', 'n', 'node'),
    ('İstanbul', 'i̇stanbul', 'İstanbul', '_stanbul', 'stanbul', 'stanbul'),
]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: repr(r[0]))
def test_safe_var_name_is_unchanged(row):
    text, lowered, kept_case = row[0], row[1], row[2]

    assert safe_var_name(text) == lowered
    assert safe_var_name(text, lowercase=False) == kept_case


@pytest.mark.parametrize("row", ROWS, ids=lambda r: repr(r[0]))
def test_naming_sanitize_is_unchanged(row):
    assert sanitize(row[0]) == row[3]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: repr(r[0]))
def test_sanitize_python_identifier_is_unchanged(row):
    assert sanitize_python_identifier(row[0], "n", set()) == row[4]
    assert sanitize_python_identifier(row[0], "node", set()) == row[5]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: repr(r[0]))
def test_every_output_is_a_python_identifier(row):
    for out in (*row[1:], safe_var_name(row[0]), safe_var_name(row[0], lowercase=False)):
        assert out.isidentifier(), out


@pytest.mark.parametrize("text, expected", [
    ("area m²", "area_m2"),          # NFKC: superscript two reads as 2
    ("x½", "x1_2"),                  # NFKC gives 1⁄2; the fraction slash is not an identifier char
    ("①", "_1"),                     # NFKC gives 1, which cannot start an identifier
    ("①⃝", "_1"),          # an enclosing mark (Me) is not an identifier char either
    ("̀abc", "_̀abc"),     # a combining mark may continue but not start one
    ("ℌilbert", "Hilbert"),          # NFKC: the parser would read ℌ as H anyway
    ("__2__4__5_OneOf", "_2_4_5_OneOf"),   # owl2uml's oneOf class names for numeric literals
    ("2 4 5", "_2_4_5"),
])
def test_safe_var_name_makes_identifiers_the_parser_accepts(text, expected):
    result = safe_var_name(text, lowercase=False)

    assert result == expected
    assert result.isidentifier()
    compile(f"{result} = 1", "<test>", "exec")


def test_the_digit_prefix_survives_stripping():
    assert sanitize_identifier("_9", strip_underscores=True) == "_9"
    assert sanitize_identifier("__9__", strip_underscores=True, digit_prefix="n_") == "n_9"


def test_sanitize_python_identifier_still_makes_names_unique():
    used = {"n", "n_2"}

    assert sanitize_python_identifier("n", "n", used) == "n_3"
    assert sanitize_python_identifier("", "n", used) == "n_4"      # falls back to the prefix, then uniquifies
    assert used == {"n", "n_2", "n_3", "n_4"}


@pytest.mark.parametrize("text", ["Café", "naïve façade", "日本語", "x²", "İstanbul", "with😀emoji"])
def test_names_that_reach_ocl_are_ascii(text):
    """The B-OCL identifier is [a-zA-Z_][a-zA-Z0-9_]*; anything else breaks self.<name>."""
    assert sanitize(text).isascii()
    assert sanitize_python_identifier(text, "n", set()).isascii()


def test_the_generated_code_names_keep_unicode_letters():
    """Code-builder names never reach OCL, and keeping the letters keeps them readable."""
    assert safe_var_name("Café", lowercase=False) == "Café"


def test_the_core_options_are_independent():
    assert sanitize_identifier("a b", ascii_only=True) == "a_b"
    assert sanitize_identifier("é", ascii_only=True) == "_"
    assert sanitize_identifier("é") == "é"
    assert sanitize_identifier("a__b") == "a__b"
    assert sanitize_identifier("a__b", collapse_underscores=True) == "a_b"
    assert sanitize_identifier("_a_", strip_underscores=True) == "a"
    assert sanitize_identifier("___", strip_underscores=True, empty="x") == "x"
    assert sanitize_identifier("", empty="x") == "x"
    assert sanitize_identifier("1a", digit_prefix="id_") == "id_1a"
    assert sanitize_identifier("AbC", lowercase=True) == "abc"
    assert sanitize_identifier("class") == "class"
    assert sanitize_identifier("class", escape_keywords=True) == "class_"
