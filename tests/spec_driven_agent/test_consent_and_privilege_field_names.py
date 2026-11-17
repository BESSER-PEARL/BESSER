"""Which required register-form fields the constructibility probe fills for its own account.

Review probes on PR 788 (commit 1ef4ab0e): ``is_marketing_user``, ``legal_user`` or
``is_cookie`` were ticked as consent boxes while ``service_terms`` or ``userAgreement``
refused registration; ``terms_админ`` lost its Cyrillic word and was ticked; a required
``group: int`` was sent as 1, the seeded staff group's id, like ``group_id`` before it.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation import api_probe, constructibility
from besser.spec_driven_agent.validation.constructibility import _choice

_BOOLEAN = {"type": "boolean"}
_INTEGER = {"type": "integer"}
_STRING = {"type": "string"}
_RANK = "a reference to another record or a rank"


@pytest.mark.parametrize("field", [
    "service_terms", "use_terms", "policy_terms", "terms_of_use_policy", "userAgreement",
    "privacy_policy", "terms", "agreement", "user_agreement_accepted", "accept_privacy_policy",
])
def test_a_consent_box_is_ticked(field):
    assert _choice(field, _BOOLEAN) is None


@pytest.mark.parametrize("field", [
    "is_marketing_user", "legal_user", "is_legal", "gdpr_user", "newsletter_user", "adult_user", "is_cookie",
    "terms_админ", "privacy", "use_privacy", "service_policy", "confirmed_user", "accept",
])
def test_a_box_that_reads_as_no_consent_is_a_choice(field):
    assert _choice(field, _BOOLEAN) == "a boolean value"


@pytest.mark.parametrize("field", ["admin_terms_accepted", "staffAgreement", "accept_role_terms", "is_superuser"])
def test_a_privilege_word_refuses_a_box_named_like_consent(field):
    assert _choice(field, _BOOLEAN) == "a privilege-named field"


# Review probe, round 6: a name opening with a non-ASCII letter swallowed the camelCase words after
# it (``überAdmin`` read as one word), so the probe registered itself with ``überAdmin: 1``.
_ACCENTED_PRIVILEGES = ["überAdmin", "créerRole", "isÉlevatedAdmin", "agréeAdmin", "écoleAdminLevel",
                        "réinitialisationAdmin"]


@pytest.mark.parametrize("field", _ACCENTED_PRIVILEGES)
@pytest.mark.parametrize("schema", [_BOOLEAN, _INTEGER, _STRING])
def test_a_privilege_word_after_an_accented_one_still_refuses(field, schema):
    assert _choice(field, schema) == "a privilege-named field"


def test_unicode_letters_make_their_own_word():
    assert constructibility._name_tokens("terms_админ") == ["terms", "админ"]
    assert constructibility._name_tokens("isAdminПраво") == ["is", "admin", "право"]
    assert constructibility._name_tokens("überAdmin") == ["ü", "ber", "admin"]


def test_the_probe_and_test_api_split_names_with_one_tokeniser():
    assert constructibility._name_tokens is api_probe.name_words


@pytest.mark.parametrize("field, schema, reason", [
    # A number naming a group or an access grant references a record or a level.
    ("group", _INTEGER, _RANK), ("user_group", _INTEGER, _RANK), ("access", _INTEGER, _RANK),
    ("authority", _INTEGER, _RANK), ("access_rights", _INTEGER, _RANK),
    # Free text cannot equal a privilege the app checks for; a discriminator number is a code.
    ("group", _STRING, None), ("access", _STRING, None), ("authority", _STRING, None), ("rights", _STRING, None),
    ("type", _STRING, None), ("kind", _STRING, None), ("type", _INTEGER, None), ("kind", _INTEGER, None),
    ("group_size", _INTEGER, None), ("access_count", _INTEGER, None),
])
def test_a_number_naming_a_group_or_access_is_a_choice(field, schema, reason):
    assert _choice(field, schema) == reason
