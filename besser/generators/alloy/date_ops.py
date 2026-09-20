"""
date_ops.py

Registry of OCL Date comparison operations (``(ocl_name, alloy_name, alloy_code)``
tuples) and ``date.als`` module generation, plus shared date helpers
(``DATES_DICT``, ``parse_ocl_date``, ``encode_date``, ``random_date`` and
``generate_dates_and_order``).

The ``date.als`` module mirrors ``strings.als``: it owns the ``Date`` signature
and the default comparison predicates (``dateGt``/``dateGte``/``dateLt``/``dateLte``),
which are named to avoid clashing with ``util/ordering``'s own ``gt/gte/lt/lte``.
"""

import random
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar

_YEAR_START = 1970

_YEAR_END = 2038

DATES_DICT: dict[str, str] = {}


def parse_ocl_date(s: str) -> date:
    """Converts ``'dMMDDYYYY'`` -> ``date``. E.g. ``'d10131977'`` -> ``date(1977, 10, 13)``."""
    mm = int(s[1:3])
    dd = int(s[3:5])
    yyyy = int(s[5:9])
    return date(yyyy, mm, dd)


def encode_date(d: date) -> str:
    """Converts ``date`` -> ``'dMMDDYYYY'``."""
    return 'd' + d.strftime('%m%d%Y')


def random_date(start: date, end: date) -> date:
    """Generates a random date between *start* and *end* (inclusive)."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def generate_dates_and_order(
    ocl_dates: list[str],
    scope: int,
    start: date = date(_YEAR_START, 1, 1),
    end: date = date(_YEAR_END, 1, 1),
    max_attempts: int = 10000,
) -> str:
    """
    Fills *ocl_dates* up to *scope* with new unique dates, emits a
    ``one sig DateN extends Date {}`` line for every date (sorted ascending),
    then appends a fact fixing the total order of all dates from smallest to
    largest.

    """
    dates_set = set(ocl_dates)
    res = ''

    attempts = 0
    while len(dates_set) < scope:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not generate a unique new date after {max_attempts} attempts "
                f"(date range may be exhausted for scope={scope})."
            )
        new_d = random_date(start, end)
        encoded = encode_date(new_d)

        # skip if already present, retry
        if encoded in dates_set:
            attempts += 1
            continue

        dates_set.add(encoded)
        attempts = 0  # reset counter after a successful generation

    # sort all dates (original + generated) ascending
    sorted_dates = sorted(dates_set, key=parse_ocl_date)

    # Rebuild DATES_DICT with sequential sig names and emit one sig per date.
    DATES_DICT.clear()
    for i, d in enumerate(sorted_dates):
        DATES_DICT[f"Date{i}"] = d
        res += f"one sig Date{i} extends Date {{}}\n"

    # build ordering fact using util/ordering's first/last/next
    date_names = [f"Date{i}" for i in range(len(sorted_dates))]
    fact_lines = [f'{date_names[0]} = first']
    for i in range(len(date_names) - 1):
        fact_lines.append(f'{date_names[i]}.next = {date_names[i + 1]}')
    fact_lines.append(f'{date_names[-1]} = last')

    res += 'fact DateOrder {\n'
    res += '\n'.join(f'    {line}' for line in fact_lines)
    res += '\n}\n'

    return res


DateOp = tuple[str, str, str]


class DateOpError(ValueError):
    """Raised when an OCL Date comparison operation is not recognised."""


class DateOpsRegistry:
    """Registry of OCL Date comparison operations (3-tuples) and ``date.als`` generator."""

    _DEFAULT_OPERATIONS: ClassVar[list[DateOp]] = [
        (">=", "dateGte", "pred dateGte[a, b: Date] { a = b or a in nexts[b] }"),
        ("<=", "dateLte", "pred dateLte[a, b: Date] { a = b or a in prevs[b] }"),
        (">",  "dateGt",  "pred dateGt[a, b: Date] { a in nexts[b] }"),
        ("<",  "dateLt",  "pred dateLt[a, b: Date] { a in prevs[b] }"),
    ]

    def __init__(self, operations: Iterable[DateOp] | None = None) -> None:
        self._ops: dict[str, DateOp] = {}
        for ocl_name, alloy_name, alloy_code in (
            operations if operations is not None else self._DEFAULT_OPERATIONS
        ):
            self.register(ocl_name, alloy_name, alloy_code)

    def register(self, ocl_name: str, alloy_name: str, alloy_code: str) -> None:
        """Maps the OCL operation *ocl_name* to the Alloy callable *alloy_name*
        whose Alloy definition is *alloy_code*."""
        self._ops[ocl_name.lower()] = (ocl_name, alloy_name, alloy_code)

    def registered_names(self) -> list[str]:
        """Returns the sorted list of registered operation names."""
        return sorted(self._ops)

    def translate(self, op: str, left: str, right: str) -> str | None:
        """Translates the binary comparison *op* on the translated operands
        *left*/*right* to the corresponding Alloy call, or returns ``None`` when
        the operation is not registered."""
        entry = self._ops.get(op.lower())
        if entry is None:
            return None
        _, alloy_name, _ = entry
        return f"({alloy_name}[{left},{right}])"

    def generate_date_block(self, ocl_dates: list[str], scope: int) -> str:
        """Returns the ``one sig DateN`` declarations and ``fact DateOrder`` for
        the given *ocl_dates*, filling them up to *scope*."""
        return generate_dates_and_order(ocl_dates, scope)

    def generate_date_ops_model(self, output_dir: str | Path, date_block: str = "") -> Path:
        """Writes ``date.als`` in *output_dir* with the ``Date`` signature, every
        registered snippet and the per-model *date_block* (``one sig DateN``
        declarations plus ``fact DateOrder``).

        ``model.als`` opens this module via ``open date``, so it must live in
        the same directory as the generated specification.
        """
        snippets = "\n\n".join(entry[2] for entry in self._ops.values())
        content = (
            "module date\n"
            "open util/boolean\n"
            "open util/ordering[Date]\n"
            "\n"
            "sig Date {}\n"
        )
        if snippets:
            content += "\n" + snippets + "\n"
        if date_block:
            content += "\n" + date_block
        path = Path(output_dir) / "date.als"
        path.write_text(content, encoding="utf-8")
        return path