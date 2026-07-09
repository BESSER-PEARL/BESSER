"""Build/runtime artifacts must never reach the download zip or a GitHub push.

Regression: a pushed repo (ArmenSl/my_axxxx) ended up committing the run's own
download zip (``besser_smart_*.zip``) and the SQLite DB that seed_data creates
(``*.db``). ``_EXCLUDED_OUTPUT_DIRS`` — fed to ``shutil.ignore_patterns`` by
BOTH the modify/continue-from-repo seed copy and the push copy — listed build
*directories* but no artifact *file* globs, so those files sailed through.
"""
import os
import shutil
import tempfile

from besser.utilities.web_modeling_editor.backend.services.smart_generation.runner import (
    _EXCLUDED_OUTPUT_DIRS,
)


def _copy_with_exclusions(files, dirs):
    src = tempfile.mkdtemp()
    for name in files:
        with open(os.path.join(src, name), "w", encoding="utf-8") as fh:
            fh.write("x")
    for d in dirs:
        os.makedirs(os.path.join(src, d))
        with open(os.path.join(src, d, "inner"), "w", encoding="utf-8") as fh:
            fh.write("x")
    dst = os.path.join(tempfile.mkdtemp(), "out")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*_EXCLUDED_OUTPUT_DIRS))
    return sorted(os.listdir(dst))


def test_run_artifacts_are_excluded():
    kept = _copy_with_exclusions(
        files=[
            "besser_smart_abc123.zip",   # the run's own download artifact
            "luxembourg.db",             # SQLite runtime DB from seed_data
            "app.sqlite3",
            "trace.log",
            "main_api.py",               # real source — must survive
            "README.md",
        ],
        dirs=["node_modules", "__pycache__", ".git"],
    )
    assert kept == ["README.md", "main_api.py"]


def test_source_files_are_not_over_excluded():
    # A ".py" that merely contains "db" in its name must NOT be dropped —
    # the globs are anchored to extensions, not substrings.
    kept = _copy_with_exclusions(
        files=["database.py", "models.py", "seed_data.py"],
        dirs=[],
    )
    assert kept == ["database.py", "models.py", "seed_data.py"]
