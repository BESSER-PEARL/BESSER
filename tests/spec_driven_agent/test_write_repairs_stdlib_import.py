"""A write that forgot ``import re`` gets it added, never ``re`` on the typing line.

Live run r12-p81: the repair turned ``from typing import List, Optional`` into
``from typing import List, Optional, re``; the module imported and failed at runtime.
"""

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

SOURCE = (
    "from typing import List, Optional\n\n\n"
    "def words(text: Optional[str]) -> List[str]:\n"
    "    return re.findall(r'\w+', text or '')\n"
)


def test_a_written_file_missing_a_stdlib_import_gets_import_re(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    content, notes = executor._repair_imports("backend/text_rules.py", SOURCE)
    assert "import re\n" in content
    assert "from typing import List, Optional\n" in content
    assert notes
