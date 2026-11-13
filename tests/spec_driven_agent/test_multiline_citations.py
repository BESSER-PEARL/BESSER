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


# Live (2026-10-30, agent judge): a line holding a "\\n" string was cut at it, so
# its call db.query(Task).all() could not be cited, and an invented tail after
# the escape matched the real line.
_JOINED = 'def listing_body(session):\n    session.reply("\\n".join(t.title for t in db.query(Task).all()) or "No tasks.")\n'


def _check_joined(tmp_path, quote):
    (tmp_path / "bot.py").write_text(_JOINED, encoding="utf-8")
    return verify_evidence([_verdict(f"bot.py: {quote}", text="lists tasks", kind="agent_data")], str(tmp_path))[0]


def test_a_call_after_an_escaped_newline_verifies(tmp_path):
    verdict = _check_joined(tmp_path, "db.query(Task).all()")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_whole_line_holding_an_escaped_newline_verifies(tmp_path):
    quote = 'session.reply("\\n".join(t.title for t in db.query(Task).all()) or "No tasks.")'
    verdict = _check_joined(tmp_path, quote)
    assert verdict["status"] == "implemented", verdict.get("note")


def test_an_invented_tail_after_an_escaped_newline_is_refused(tmp_path):
    verdict = _check_joined(tmp_path, 'session.reply("\\n".join(t.title for t in db.query(Task).filter(done=False)))')
    assert verdict["status"] == "unverified"


# Live (2026-10-31, Android run): the judge cut a Kotlin data class as
# "data class TodoItemEntity(..., val listId: Long)" - an ellipsis then a comma.
_ENTITY_KT = (
    "data class TodoItemEntity(\n"
    "    @PrimaryKey(autoGenerate = true) val id: Long = 0,\n"
    "    val title: String,\n"
    "    val listId: Long\n"
    ") {\n"
    "}\n"
)


def _check_entity_kt(tmp_path, quote):
    (tmp_path / "Entities.kt").write_text(_ENTITY_KT, encoding="utf-8")
    return verify_evidence([_verdict(f"Entities.kt: {quote}", text="Organize tasks into lists", kind="")],
                           str(tmp_path))[0]


def test_an_ellipsis_followed_by_a_comma_cuts_the_quote(tmp_path):
    verdict = _check_entity_kt(tmp_path, "data class TodoItemEntity(..., val listId: Long)")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_an_invented_field_after_an_ellipsis_and_comma_is_refused(tmp_path):
    verdict = _check_entity_kt(tmp_path, "data class TodoItemEntity(..., val ownerId: Long)")
    assert verdict["status"] == "unverified"


# Live (2026-11-13, claude-sonnet-5-5, wastewater dashboard): the only blockers
# left were these two correct citations of React code, both rejected.
MAP_JSX = '''/**
 * onSelectRegion(regionName) fires on click.
 */
export default function MapView({ regionsGeo, onSelectRegion }) {
  const selectRef = useRef(onSelectRegion);
  const layer = L.geoJSON(regionsGeo, {
    onEachFeature: (feature, lyr) => {
      const name = featureRegion(feature);
      lyr.on('click', () => name && selectRef.current && selectRef.current(name));
    },
  }).addTo(map);
}
'''
DASHBOARD_JSX = '''export default function Dashboard() {
  return (
    <div>
      <MapView regionsGeo={regions} onSelectRegion={setSelected} />
      <RegionPlot region={selected} />
      {/* Key metrics */}
      <section className="kpis">
        <Kpi label="Treatment plants" value={fmt(summary?.plant_count, 0)} hint={`${fmt(summary?.plants_with_detections, 0)} with data`} />
        <Kpi label="Viruses tracked" value={fmt(summary?.virus_count, 0)} />
        <Kpi label="Detections" value={fmt(summary?.detection_count, 0)} hint={summary?.first_sample_date ? `${summary.first_sample_date}` : 'no samples yet'} />
        <Kpi label="Avg. concentration" value={fmt(summary?.avg_concentration)} />
        <Kpi
          label="Peak concentration"
          value={fmt(summary?.max_concentration)}
          hint={summary?.peak_detection ? `${summary.peak_detection.virus}` : undefined}
        />
        <Kpi label="Latest sample" value={summary?.latest_sample_date || '-'} />
      </section>
    </div>
  );
}
'''


def _check_jsx(tmp_path, evidence, text="Show the plot of a clicked region", kind="action"):
    (tmp_path / "frontend" / "src" / "components").mkdir(parents=True, exist_ok=True)
    (tmp_path / "frontend" / "src" / "pages").mkdir(exist_ok=True)
    (tmp_path / "frontend" / "src" / "components" / "MapView.jsx").write_text(MAP_JSX, encoding="utf-8")
    (tmp_path / "frontend" / "src" / "pages" / "Dashboard.jsx").write_text(DASHBOARD_JSX, encoding="utf-8")
    return verify_evidence([_verdict(evidence, text=text, kind=kind)], str(tmp_path))[0]


def test_two_files_after_a_semicolon_with_a_trailing_explanation_verify(tmp_path):
    evidence = ("frontend/src/components/MapView.jsx: lyr.on('click', () => name && selectRef.current && "
                "selectRef.current(name)); and frontend/src/pages/Dashboard.jsx: <RegionPlot region={selected} /> "
                "which calls fetchDetectionsByRegion(region) and renders LineChart with detection data for that region")
    verdict = _check_jsx(tmp_path, evidence)
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_jsx_block_cut_with_an_ellipsis_inside_its_last_line_verifies(tmp_path):
    evidence = ('frontend/src/pages/Dashboard.jsx: <section className="kpis"> <Kpi label="Treatment plants" '
                'value={fmt(summary?.plant_count, 0)} hint={`${fmt(summary?.plants_with_detections, 0)} with data`} /> '
                '<Kpi label="Viruses tracked" value={fmt(summary?.virus_count, 0)} /> <Kpi label="Detections" '
                'value={fmt(summary?.detection_count, 0)} ... <Kpi label="Avg. concentration" '
                'value={fmt(summary?.avg_concentration)} /> <Kpi label="Peak concentration" '
                'value={fmt(summary?.max_concentration)} ... <Kpi label="Latest sample" '
                "value={summary?.latest_sample_date || '-'} />")
    verdict = _check_jsx(tmp_path, evidence, text="Display key metrics on the dashboard", kind="ui")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_jsx_citations_still_refuse_invented_misfiled_and_comment_quotes(tmp_path):
    for evidence in (
        # invented code, with and without a trailing explanation
        "frontend/src/components/MapView.jsx: lyr.on('click', () => openPlot(name)) which shows the plot",
        "frontend/src/pages/Dashboard.jsx: <RegionPlot region={selected} showChart /> which renders a chart",
        # a real line cited from the wrong file, after a real part
        "frontend/src/components/MapView.jsx: const selectRef = useRef(onSelectRegion); and "
        "frontend/src/components/MapView.jsx: <RegionPlot region={selected} /> which renders the plot",
        # a comment that only mentions the behaviour
        "frontend/src/components/MapView.jsx: * onSelectRegion(regionName) fires on click.",
        # a block whose cut line is not the next source line
        'frontend/src/pages/Dashboard.jsx: <section className="kpis"> <Kpi label="Detections" value={fmt( ...',
        # an explanation not set off from the code is still prose
        "frontend/src/pages/Dashboard.jsx: RegionPlot which renders the detections of the selected region",
    ):
        assert _check_jsx(tmp_path, evidence)["status"] == "unverified", evidence


# Live (2026-11-13, claude-sonnet-5-5, wastewater dashboard): "; and" joined two
# quotes of one file and the "and" was read as code ("andL.tileLayer(").
LEAFLET_JSX = '''export default function MapView() {
  useEffect(() => {
    const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
    }).addTo(map);
  }, []);
}
'''
SUMMARY_PY = '''def get_summary(database):
    n_plants = database.query(TreatmentPlant).count()
    return {"plants": n_plants}
'''


def _check_two_files(tmp_path, evidence):
    (tmp_path / "frontend" / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "backend" / "routers").mkdir(parents=True, exist_ok=True)
    (tmp_path / "frontend" / "src" / "MapView.jsx").write_text(LEAFLET_JSX, encoding="utf-8")
    (tmp_path / "backend" / "routers" / "dashboard.py").write_text(SUMMARY_PY, encoding="utf-8")
    return verify_evidence([_verdict(evidence, text="Show a map with plant counts", kind="ui")], str(tmp_path))[0]


def test_and_after_a_semicolon_joins_two_quotes_of_one_file(tmp_path):
    evidence = ("frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
                "and L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {")
    verdict = _check_two_files(tmp_path, evidence)
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_file_path_after_an_ellipsis_switches_the_file(tmp_path):
    evidence = ("frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
                "... backend/routers/dashboard.py: n_plants = database.query(TreatmentPlant).count()")
    verdict = _check_two_files(tmp_path, evidence)
    assert verdict["status"] == "implemented", verdict.get("note")


def test_joined_quotes_still_refuse_an_invented_or_misfiled_part(tmp_path):
    for evidence in (
        # invented code after "; and"
        "frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
        "and L.marker(PLANT).addTo(map).openPopup();",
        # a real Python line cited under the JSX file
        "frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
        "and n_plants = database.query(TreatmentPlant).count()",
        # a path that names no judged file does not switch: the line is not in MapView.jsx
        "frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
        "... backend/routers/missing.py: n_plants = database.query(TreatmentPlant).count()",
        # an elided signature glued to a return statement lines later
        "frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
        "... backend/routers/dashboard.py: def get_summary(...) return {\"plants\": n_plants, \"viruses\": n}",
    ):
        assert _check_two_files(tmp_path, evidence)["status"] == "unverified", evidence


# Live (2026-11-13, claude-sonnet-5-5, risk-awareness app): a verbatim citation of
# a comprehension was rejected because a middle line ends in "# noqa: E712".
ENGINE_PY = '''def detect(db, account, since):
    unwanted = [
        m
        for m in db.query(Message)
        .filter(Message.sender_id == account.id, Message.isUnwanted == True)
        .all()  # noqa: E712
        if m.sentAt >= since and m.channel != "#general"
    ]
    return unwanted
'''
UNWANTED = ("unwanted = [ m for m in db.query(Message) .filter(Message.sender_id == account.id, "
            "Message.isUnwanted == True) .all() if m.sentAt >= since and m.channel != \"#general\" ]")
STATS_JSX = '''export default function Stats({ s }) {
  return (
    <section>
      <Kpi label="Signals" value={s.signals} />
      {/* risk summary */}
      <Kpi label="Risk // level" value={s.level} />
    </section>
  );
}
'''


def _check_engine(tmp_path, evidence, text="Detect repeated unwanted messages", kind="computed"):
    (tmp_path / "backend").mkdir(exist_ok=True)
    (tmp_path / "frontend").mkdir(exist_ok=True)
    (tmp_path / "backend" / "risk_engine.py").write_text(ENGINE_PY, encoding="utf-8")
    (tmp_path / "frontend" / "Stats.jsx").write_text(STATS_JSX, encoding="utf-8")
    return verify_evidence([_verdict(evidence, text=text, kind=kind)], str(tmp_path))[0]


def test_a_multi_line_quote_over_a_trailing_comment_verifies(tmp_path):
    verdict = _check_engine(tmp_path, f"backend/risk_engine.py: {UNWANTED}")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_multi_line_jsx_quote_over_a_comment_line_verifies(tmp_path):
    evidence = ('frontend/Stats.jsx: <Kpi label="Signals" value={s.signals} /> '
                '<Kpi label="Risk // level" value={s.level} />')
    verdict = _check_engine(tmp_path, evidence, text="Show the risk level", kind="ui")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_comments_are_not_dropped_from_strings_or_accepted_alone(tmp_path):
    for evidence, kind in (
        # a "#" inside a string is not a comment: the string cannot be cut there
        ("backend/risk_engine.py: " + UNWANTED.replace('"#general"', '"'), "computed"),
        # the comment alone, and the comment as a line's substance
        ("backend/risk_engine.py: # noqa: E712", "computed"),
        ("backend/risk_engine.py: .all() # noqa: E712 must be unwanted", "computed"),
        ("frontend/Stats.jsx: {/* risk summary */}", "ui"),
        # a quote whose comment-free text was invented
        ("backend/risk_engine.py: " + UNWANTED.replace(".all()", ".limit(5)"), "computed"),
    ):
        assert _check_engine(tmp_path, evidence, kind=kind)["status"] == "unverified", evidence


# Live (2026-11-13, claude-sonnet-5-5): the judge stitched separate JSX lines into
# one quote, or glued an elided signature to a return lines later; the rejection
# note did not say which part failed, so the re-judge could not correct it.
def test_a_rejected_citation_names_the_part_that_matched_nothing(tmp_path):
    evidence = ("frontend/src/MapView.jsx: const map = L.map(elRef.current, { center: LUX_CENTER, zoom: 9 }); "
                "... backend/routers/dashboard.py: def get_summary(...) return {\"plants\": n_plants, \"viruses\": n}")
    note = _check_two_files(tmp_path, evidence)["note"]
    assert 'no source line matches the part `) return {"plants": n_plants, "viruses": n}`' in note


def test_the_named_part_is_clipped(tmp_path):
    evidence = "frontend/src/MapView.jsx: " + "<>...<Metric title=\"Detections\" value={metrics.count} />" * 4
    note = _check_two_files(tmp_path, evidence)["note"]
    part = note.split("matches the part `", 1)[1].split("`", 1)[0]
    assert part.startswith("<>...<Metric title=\"Detections\"") and part.endswith("...") and len(part) == 80


def test_the_judge_is_told_to_cite_lines_as_separate_parts():
    from besser.spec_driven_agent.planning.requirements_ledger import _JUDGE_SYSTEM_PROMPT
    assert "never join separate lines" in _JUDGE_SYSTEM_PROMPT
    assert 'separate "file: line" parts joined by " and "' in _JUDGE_SYSTEM_PROMPT
