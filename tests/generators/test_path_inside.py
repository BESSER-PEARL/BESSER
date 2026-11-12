"""``path_inside``: a file name built from a model's names stays in the output folder."""
import os

import pytest

from besser.generators.generator_interface import UnsafeOutputPathError, path_inside


@pytest.mark.parametrize("name", ["Book.java", "routers/book.py", "a/../b.py", "My Agent.py"])
def test_a_name_inside_the_folder_is_joined(tmp_path, name):
    assert path_inside(str(tmp_path), name) == os.path.join(str(tmp_path), name)


@pytest.mark.parametrize("name", ["../x.py", "../../outside/pwnagent.py", "a/../../x.py", "..", os.path.abspath("x")])
def test_a_name_that_leaves_the_folder_is_refused(tmp_path, name):
    with pytest.raises(UnsafeOutputPathError, match="leads out of the output folder"):
        path_inside(str(tmp_path), name)
