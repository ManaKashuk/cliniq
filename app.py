# =========================
# FILE: tests/test_app.py
# =========================
import os
from pathlib import Path
import importlib
import pytest

def write_file(dirpath: Path, name: str, text: str):
    p = dirpath / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p

@pytest.fixture(scope="module")
def app_module(tmp_path_factory, monkeypatch):
    sopdir = tmp_path_factory.mktemp("sops")
    monkeypatch.setenv("SOP_DIR", str(sopdir))
    # Import the app module (UI won't run due to __main__ guard)
    mod = importlib.import_module("app")
    return mod

def test_load_documents_placeholder_when_empty(app_module):
    docs = app_module.load_documents(app_module.DATA_DIR)
    assert docs and "No SOP files found" in docs[0][1]

def test_tfidf_retrieval_returns_snippets(app_module, tmp_path, monkeypatch):
    sopdir = Path(os.environ["SOP_DIR"])
    write_file(sopdir, "gcp_sop.txt", "AE reporting requires 24-hour notification for SAEs to sponsor.")
    write_file(sopdir, "ip_sop.txt", "IP shipment and chain of custody for CRC. Temperature 2–8°C.")
    docs = app_module.load_documents(sopdir)
    vec, mat, sources, corpus = app_module.build_tfidf_index(docs)
    query = app_module.build_query("CRC", "IP shipment", {"Temperature control?": "Refrigerated (2–8°C)"})
    snips = app_module.retrieve(query, vec, mat, sources, corpus, k=3)
    assert snips and any("ip" in s.source.lower() for s in snips)

def test_offline_compose_schema(app_module):
    snips = [app_module.Snippet(text="x", source="ip_sop.txt", score=0.9)]
    plan = app_module.offline_compose("Clinical Research Coordinator (CRC)", "Missed visit", {}, snips)
    for k in ["steps", "required_docs", "escalations", "citations", "compliance", "disclaimer"]:
        assert k in plan
    assert plan["disclaimer"] == app_module.FINAL_VERIFICATION_LINE

def test_markdown_contains_sections(app_module):
    plan = {
        "steps": ["s1", "s2"],
        "required_docs": ["d1"],
        "escalations": ["e1"],
        "citations": ["Source: a.txt"],
        "compliance": ["c1"],
        "disclaimer": app_module.FINAL_VERIFICATION_LINE,
    }
    md = app_module.to_markdown("Clinical Research Coordinator (CRC)", "Protocol deviation", {}, plan)
    assert "## Steps" in md and "## SOP Citations" in md and app_module.FINAL_VERIFICATION_LINE in md

def test_pdf_export_fallback(app_module):
    out = app_module.to_pdf_bytes("# Title\nBody")
    assert (out is None) or (isinstance(out, (bytes, bytearray)) and len(out) > 0)
