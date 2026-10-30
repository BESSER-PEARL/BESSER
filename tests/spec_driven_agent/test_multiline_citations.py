"""A correct citation of formatted code is accepted; an invented one is not.

Live evidence (2026-10-30, claude-sonnet-5-5, a local Windows 11 todo app):
every desktop run ended with 4-6 "requirement unverified" blockers on code
that did implement the requirement. The judge quotes a statement on one line,
the source spreads it over several (a formatted call, two adjacent string
pieces, a def and its first body line), and the checker only compared single
lines. It also abridges a run as "a ... b" or cites two places joined by "and".
"""

from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence

DB = '''import sqlite3


class Repository:
    def create_item(self, title, list_id):
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO todo_item(title, list_id) "
                "VALUES (?,?)", (title, list_id))
        return cur.lastrowid

    def items_grouped_by_list(self, list_id=None, tag_ids=(),
                              match_all=False):
        return [list_id, tag_ids, match_all]

    def new_task(self):
        if not self.lists:
            return None
        res = self.dialog().show()
        return res
'''


def _verdict(evidence, text="Create a new task", kind="action"):
    return {"id": 1, "kind": kind, "text": text, "status": "implemented", "evidence": evidence}


def _check(tmp_path, evidence, **kwargs):
    (tmp_path / "todos").mkdir(exist_ok=True)
    (tmp_path / "todos" / "db.py").write_text(DB, encoding="utf-8")
    return verify_evidence([_verdict(evidence, **kwargs)], str(tmp_path))[0]


def test_a_formatted_call_quoted_on_one_line_verifies(tmp_path):
    verdict = _check(tmp_path, 'todos/db.py: cur = self.conn.execute("INSERT INTO todo_item(title, list_id) '
                               'VALUES (?,?)", (title, list_id))')
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_signature_quoted_without_its_tail_verifies(tmp_path):
    verdict = _check(tmp_path, "todos/db.py: def items_grouped_by_list(self, list_id=None, tag_ids=(), "
                               "match_all=False)", text="View tasks grouped by list", kind="ui")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_an_abridged_run_verifies_on_its_first_part(tmp_path):
    verdict = _check(tmp_path, "todos/db.py: with self.conn: cur = self.conn.execute( ... return cur.lastrowid")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_two_joined_citations_verify_when_both_exist(tmp_path):
    (tmp_path / "todos").mkdir()
    (tmp_path / "todos" / "ui.py").write_text("def main():\n    App().mainloop()\n", encoding="utf-8")
    verdict = _check(tmp_path, "todos/db.py: res = self.dialog().show() and todos/ui.py: App().mainloop()",
                     text="Open the task dialog", kind="ui")
    assert verdict["status"] == "implemented", verdict.get("note")


# Review of this change: a real part must never carry an invented one.
def test_an_invented_part_rejects_the_whole_citation(tmp_path):
    for evidence in (
        "todos/db.py: return cur.lastrowid; self.conn.execute(\"DELETE FROM todo_item WHERE id=?\", (id,))",
        "todos/db.py: return cur.lastrowid ... raise ValueError(\"duplicate title\")",
        "todos/db.py: res = self.dialog().show() and todos/missing.py: App().mainloop()",
    ):
        assert _check(tmp_path, evidence)["status"] == "unverified", evidence


def test_text_inside_a_comment_or_string_is_not_code(tmp_path):
    (tmp_path / "todos").mkdir(exist_ok=True)
    (tmp_path / "todos" / "svc.py").write_text(
        "def add(title):\n"
        "    save(title)  # must be unique, raise ValueError on duplicate title\n"
        "    print(\"validates that the due date is after the start date\")\n",
        encoding="utf-8")
    for quote in ("must be unique, raise ValueError on duplicate title",
                  "validates that the due date is after the start date"):
        verdict = verify_evidence([_verdict(f"todos/svc.py: {quote}", kind="rule", text="Titles are unique")],
                                  str(tmp_path))[0]
        assert verdict["status"] == "unverified", quote


def test_a_quote_that_skips_lines_is_still_rejected(tmp_path):
    # The judge left out the guard clause without marking the gap.
    verdict = _check(tmp_path, "todos/db.py: def new_task(self): res = self.dialog().show()")
    assert verdict["status"] == "unverified"


def test_an_invented_quote_is_still_rejected(tmp_path):
    verdict = _check(tmp_path, 'todos/db.py: cur = self.conn.execute("DELETE FROM todo_item")')
    assert verdict["status"] == "unverified"
    assert "not found in the cited source" in verdict["note"]


def test_a_quote_that_drops_a_statements_substance_is_rejected(tmp_path):
    # One full line plus a scrap of the next proves nothing about the statement.
    verdict = _check(tmp_path, 'todos/db.py: cur = self.conn.execute( "INS')
    assert verdict["status"] == "unverified"


# Live evidence, same day, native Android and desktop+agent runs (Kotlin/Compose):
# one argument per line, a block opener left off, "a; b" lists, ",...)" cuts.
KT = '''@Composable
fun ItemEditor(vm: TodoViewModel) {
    Button(onClick = { save() }) {
        Text("Save")
    }
    vm.saveItem(
        TodoItem(
            id = itemId,
            title = title.trim(),
            description = description.trim(),
            dueDate = dueDate,
            priority = priority,
            status = status,
            listId = effectiveListId,
        ),
        tagIds,
        onClose
    )
    NameDialog(
        title = "Nueva lista",
        onConfirm = { n -> vm.saveList(0, n); creating = false },
        onDismiss = { creating = false }
    )
}

fun createTag(name: String): Tag = repo.insertTag(name)
fun deleteTag(id: Long) = repo.deleteTag(id)
'''


def _check_kt(tmp_path, evidence):
    (tmp_path / "ui").mkdir(exist_ok=True)
    (tmp_path / "ui" / "Screen.kt").write_text(KT, encoding="utf-8")
    return verify_evidence([_verdict(evidence, kind="ui", text="Edit a task")], str(tmp_path))[0]


def test_a_long_kotlin_call_with_a_trailing_comma_verifies(tmp_path):
    verdict = _check_kt(tmp_path, "ui/Screen.kt: vm.saveItem(TodoItem(id = itemId, title = title.trim(), "
                                  "description = description.trim(), dueDate = dueDate, priority = priority, "
                                  "status = status, listId = effectiveListId), tagIds, onClose)")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_line_quoted_without_the_block_it_opens_verifies(tmp_path):
    assert _check_kt(tmp_path, "ui/Screen.kt: Button(onClick = { save() })")["status"] == "implemented"


def test_a_quote_cut_with_comma_dots_verifies_on_its_first_part(tmp_path):
    verdict = _check_kt(tmp_path, 'ui/Screen.kt: NameDialog(title = "Nueva lista", onConfirm = '
                                  '{ n -> vm.saveList(0, n); creating = false },...)')
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_semicolon_list_of_lines_verifies(tmp_path):
    verdict = _check_kt(tmp_path, "ui/Screen.kt: fun createTag(name: String): Tag = repo.insertTag(name); "
                                  "fun deleteTag(id: Long) = repo.deleteTag(id)")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_line_missing_more_than_its_block_opener_is_rejected(tmp_path):
    assert _check_kt(tmp_path, "ui/Screen.kt: Button(onClick")["status"] == "unverified"


# Live, round 6 (Flutter, React Native): a complete call quoted from inside its line.
DART = '''class ListsTab extends StatelessWidget {
  Widget build(BuildContext context) {
    return FloatingActionButton(onPressed: () => showListDialog(context, store), child: Icon(Icons.add));
  }
}
'''


def _check_dart(tmp_path, evidence):
    (tmp_path / "lib").mkdir(exist_ok=True)
    (tmp_path / "lib" / "screens.dart").write_text(DART, encoding="utf-8")
    return verify_evidence([_verdict(evidence, kind="ui", text="Create a list")], str(tmp_path))[0]


def test_a_complete_call_inside_a_line_verifies(tmp_path):
    assert _check_dart(tmp_path, "lib/screens.dart: showListDialog(context, store)")["status"] == "implemented"


def test_a_call_fragment_or_a_generic_call_is_rejected(tmp_path):
    assert _check_dart(tmp_path, "lib/screens.dart: showListDialog(context")["status"] == "unverified"
    assert _check_dart(tmp_path, "lib/screens.dart: Icon(Icons)")["status"] == "unverified"


def test_prose_instead_of_code_is_rejected(tmp_path):
    # Live (SwiftUI): the judge described the code instead of quoting it.
    verdict = _check_dart(tmp_path, "lib/screens.dart: ListsTab shows lists and opens the dialog")
    assert verdict["status"] == "unverified"


# Live, round 6 (desktop): the judge quotes a line's start and marks the cut.
TK = '''class TagsPane:
    def build(self, row):
        ttk.Button(row, text="New", command=self.new_tag).pack(side="left", padx=4)
'''


def _check_tk(tmp_path, evidence):
    (tmp_path / "todos").mkdir(exist_ok=True)
    (tmp_path / "todos" / "ui.py").write_text(TK, encoding="utf-8")
    return verify_evidence([_verdict(evidence, kind="ui", text="Create a tag")], str(tmp_path))[0]


def test_a_line_start_cut_with_an_ellipsis_verifies(tmp_path):
    verdict = _check_tk(tmp_path, 'todos/ui.py: ttk.Button(row, text="New", command=self.new_tag).pack ... more')
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_short_line_start_without_an_ellipsis_is_rejected(tmp_path):
    assert _check_tk(tmp_path, 'todos/ui.py: ttk.Button(row, text="New"')["status"] == "unverified"
    assert _check_tk(tmp_path, "todos/ui.py: ttk.Button(row ... more")["status"] == "unverified"


def test_a_long_run_from_inside_a_line_verifies(tmp_path):
    # Live (Compose): the statement inside "onClick = { ... }" was quoted alone.
    verdict = _check_tk(tmp_path, 'todos/ui.py: text="New", command=self.new_tag).pack(side="left"')
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_call_whose_last_argument_has_a_trailing_comma_verifies(tmp_path):
    # Live (desktop): "(cur.lastrowid, t),\n)" - the comma on the line before the
    # closing bracket made the span look complete one line too early.
    (tmp_path / "todos").mkdir(exist_ok=True)
    (tmp_path / "todos" / "tags.py").write_text(
        "def tag(self, cur, t):\n"
        "    self.conn.execute(\n"
        "        \"INSERT INTO item_tags(item_id, tag_id) VALUES (?,?)\",\n"
        "        (cur.lastrowid, t),\n"
        "    )\n", encoding="utf-8")
    evidence = 'todos/tags.py: self.conn.execute("INSERT INTO item_tags(item_id, tag_id) VALUES (?,?)", (cur.lastrowid, t))'
    verdict = verify_evidence([_verdict(evidence)], str(tmp_path))[0]
    assert verdict["status"] == "implemented", verdict.get("note")
