"""``search_in_files`` returned third-party code instead of the app's.

The walk never pruned, and the 50-match cap is global and returns early. So
once the frontend's dependencies are installed -- which now happens inside a
run, via ``install_dependencies`` -- a search for any common identifier fills
its whole budget from ``node_modules`` and answers ``truncated: True`` without
having reached a single file the model wrote.

That is worse than returning nothing: the model gets a confident, plausible
answer about code it cannot edit and did not ask about.
"""
import pytest

from besser.generators.llm.tool_executor import ToolExecutor


def _workspace(tmp_path):
    """One real source hit, buried behind far more vendored ones."""
    app = tmp_path / "frontend" / "src"
    app.mkdir(parents=True)
    (app / "Booking.tsx").write_text(
        "export function bookingTotal() { return 0; }\n", encoding="utf-8")

    # Enough to exhaust the 50-match cap on its own, and alphabetically
    # before "frontend" so an unpruned walk reaches it first.
    vendored = tmp_path / "frontend" / "node_modules" / "acme"
    vendored.mkdir(parents=True)
    for index in range(60):
        (vendored / f"chunk{index:03d}.js").write_text(
            "exports.bookingTotal = function () { return 1; };\n", encoding="utf-8")
    return tmp_path


def test_search_does_not_return_dependency_source(tmp_path):
    executor = ToolExecutor(workspace=str(_workspace(tmp_path)))

    result = executor._search_in_files({"pattern": "bookingTotal"})
    files = {match["file"] for match in result["matches"]}

    assert not [f for f in files if "node_modules" in f], (
        "search returned installed dependency source: "
        f"{sorted(f for f in files if 'node_modules' in f)[:3]}"
    )
    assert "frontend/src/Booking.tsx" in files, (
        "the one file the model actually wrote was not reached"
    )


def test_the_match_budget_is_not_spent_before_the_app_is_reached(tmp_path):
    """The failure mode is silent: a truncated answer that looks complete."""
    executor = ToolExecutor(workspace=str(_workspace(tmp_path)))

    result = executor._search_in_files({"pattern": "bookingTotal"})

    assert not result.get("truncated"), (
        "the 50-match cap was exhausted, so the result is a partial view "
        "presented as an answer"
    )
    assert result.get("total") == 1


@pytest.mark.parametrize("vendored_dir", ["node_modules", "dist", "build", "__pycache__"])
def test_generated_and_vendored_trees_are_all_pruned(tmp_path, vendored_dir):
    """Build output is as useless to edit as a dependency, and `build/` is
    exactly where the generated Vite frontend emits."""
    (tmp_path / "app.py").write_text("marker = 1\n", encoding="utf-8")
    noise = tmp_path / vendored_dir
    noise.mkdir()
    (noise / "copy.py").write_text("marker = 2\n", encoding="utf-8")

    executor = ToolExecutor(workspace=str(tmp_path))
    files = {m["file"] for m in executor._search_in_files({"pattern": "marker"})["matches"]}

    assert files == {"app.py"}, f"{vendored_dir} was searched: {sorted(files)}"
