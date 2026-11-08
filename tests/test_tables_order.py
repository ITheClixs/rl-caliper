"""Every generated table and paper macro is the same whatever order the run records load in.

Selecting "the newest record" let later runs silently replace numbers in the paper several
times: a 7B table in the 0.5B section, a smoke run's ratio in the text, a pilot grid's summary
beside the full grid's table. This regenerates everything twice, once in the stores' natural
order and once reversed, and requires identical output. It takes minutes, so it is opt-in:
`uv run pytest -m slow tests/test_tables_order.py`.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))


@pytest.mark.slow
def test_generated_tables_do_not_depend_on_record_order(tmp_path, monkeypatch):
    import tables

    from caliper.runtime import io

    original = io.load_all
    outputs = {}
    for name, loader in (("natural", original), ("reversed", lambda s: original(s)[::-1])):
        target = tmp_path / name / "tables"
        target.mkdir(parents=True)
        monkeypatch.setattr(tables, "TABLES", target)
        monkeypatch.setattr(io, "load_all", loader)
        tables.main()
        outputs[name] = {p.name: p.read_text() for p in sorted(target.glob("*.tex"))}
    assert outputs["natural"].keys() == outputs["reversed"].keys()
    differing = [k for k in outputs["natural"] if outputs["natural"][k] != outputs["reversed"][k]]
    assert not differing, f"tables that depend on record order: {differing}"
