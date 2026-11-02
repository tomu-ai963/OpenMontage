"""Schema / checkpoint / manifest / playbook files are UTF-8 and must load under a non-UTF-8 locale.

On Windows with a Japanese locale, open() without encoding= decodes as cp932 and
fails on the em dashes in the schema descriptions. Run the loaders in a child
interpreter with UTF-8 mode off so the test reproduces that environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import tempfile
from pathlib import Path
from schemas.artifacts import ARTIFACT_NAMES, load_schema
from lib.checkpoint import write_checkpoint, read_checkpoint
from lib import pipeline_loader

for name in ARTIFACT_NAMES:
    load_schema(name)

with tempfile.TemporaryDirectory() as d:
    script = {"version": "1.0", "title": "テスト — UTF-8", "total_duration_seconds": 1,
              "sections": [{"id": "s1", "text": "神聖幾何学", "start_seconds": 0, "end_seconds": 1}]}
    write_checkpoint(Path(d), "p", "script", "completed", {"script": script}, pipeline_type="animation")
    assert read_checkpoint(Path(d), "p", "script")["artifacts"]["script"]["title"] == script["title"]

pipeline_loader.load_pipeline("animation")

import jsonschema
from styles.playbook_loader import list_playbooks, load_playbook
names = list_playbooks()
assert names
for name in names:
    try:
        load_playbook(name)
    except jsonschema.ValidationError:
        pass  # schema conformance is covered elsewhere; this test is about decoding
print("ok")
"""


def test_loaders_work_without_utf8_mode():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", SCRIPT],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")
