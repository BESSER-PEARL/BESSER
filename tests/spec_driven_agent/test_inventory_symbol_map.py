"""The inventory carries a symbol map: names with line numbers per file.

With the scaffold copy no longer pasted into the prompt, this is how the
model finds the region it needs and reads it with ``read_file`` offset/limit
instead of reading whole files (aider's repo-map idea, kept to top-level
names so it stays a few hundred tokens).
"""

from besser.spec_driven_agent.prompt_builder import build_inventory

PY = (
    "import io\n"                                   # 1
    "\n"                                            # 2
    "class BookingService:\n"                       # 3
    "    pass\n"                                    # 4
    "\n"                                            # 5
    "async def booking_cancel(database):\n"         # 6
    "    return 1\n"                                # 7
    "\n"                                            # 8
    "def helper():\n"                               # 9
    "    return 2\n"                                # 10
)
JSX = (
    "import React from 'react';\n"                  # 1
    "\n"                                            # 2
    "export default function BookingList() {\n"     # 3
    "  return null;\n"                              # 4
    "}\n"                                           # 5
    "export const useBookings = () => [];\n"        # 6
)


def test_inventory_lists_top_level_symbols_with_line_numbers(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "booking_methods.py").write_text(PY, encoding="utf-8")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "BookingList.jsx").write_text(JSX, encoding="utf-8")
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi\n", encoding="utf-8")

    inventory = build_inventory(str(tmp_path), None, "generate_fastapi_backend")

    py_line = next(ln for ln in inventory.splitlines() if "booking_methods.py" in ln)
    assert "BookingService@3" in py_line
    assert "booking_cancel@6" in py_line
    assert "helper@9" in py_line
    jsx_line = next(ln for ln in inventory.splitlines() if "BookingList.jsx" in ln)
    assert "BookingList@3" in jsx_line
    assert "useBookings@6" in jsx_line
    txt_line = next(ln for ln in inventory.splitlines() if "requirements.txt" in ln)
    assert "@" not in txt_line                       # no symbols for non-code files


def test_symbol_map_is_capped_per_file(tmp_path):
    body = "\n".join(f"def f{i}():\n    pass\n" for i in range(40))
    (tmp_path / "big.py").write_text(body, encoding="utf-8")
    inventory = build_inventory(str(tmp_path), None, "x")
    line = next(ln for ln in inventory.splitlines() if "big.py" in ln)
    assert line.count("@") <= 12
    assert "more" in line
