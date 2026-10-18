"""Check local Markdown links, fenced examples, and JSON configurations."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for directory in (ROOT, ROOT / "docs", ROOT / "native", ROOT / "research", ROOT / "checkpoints", ROOT / "legacy", ROOT / "maps", ROOT / "results"):
        paths = directory.glob("*.md") if directory == ROOT else directory.rglob("*.md")
        for path in paths:
            count += 1
            source = path.read_text()
            if source.count("```") % 2:
                raise ValueError(f"Unbalanced code fence: {path}")
            for target in re.findall(r'\]\(([^)]+)\)', source):
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                if not (path.parent / target.split("#")[0]).exists():
                    raise ValueError(f"Missing local link in {path}: {target}")
            for snippet in re.findall(r'```json\n(.*?)```', source, re.S):
                json.loads(snippet)
            for snippet in re.findall(r'```python\n(.*?)```', source, re.S):
                ast.parse(snippet)
    for path in (ROOT / "configs").rglob("*.json"):
        json.loads(path.read_text())
    print(f"Documentation checks passed ({count} Markdown files).")


if __name__ == "__main__":
    main()
