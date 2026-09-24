import pytest

from besser.utilities.path_utils import normalize_relative_path


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("data/files", "data/files"),
        ("data\\files\\", "data/files"),
        ("/tmp/sandbox", "tmp/sandbox"),
        ("C:\\Users\\me\\ws", "Users/me/ws"),
        ("c:/ws", "ws"),
        ("./a/./b", "a/b"),
        ("  a//b  ", "a/b"),
        ("//server/share", "server/share"),
    ],
)
def test_normalize_relative_path_returns_posix_relative_path(raw, expected):
    assert normalize_relative_path(raw) == expected


@pytest.mark.parametrize("raw", ["../escape", "a/../../b", "C:\\..\\x", "a/.."])
def test_normalize_relative_path_rejects_parent_segments(raw):
    with pytest.raises(ValueError, match=r"'\.\.'"):
        normalize_relative_path(raw)


@pytest.mark.parametrize("raw", ["", "   ", "/", "C:", "C:\\", "./.", None])
def test_normalize_relative_path_rejects_empty_result(raw):
    with pytest.raises(ValueError, match="relative path segment"):
        normalize_relative_path(raw)
