import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "CLINI-Q • Role-based SOP Guidance"

# Allow tests to override with SOP_DIR; default to ./data/sops
DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "sops"
ENV_SOP_DIR = os.environ.get("SOP_DIR", "").strip()
DATA_DIR = Path(ENV_SOP_DIR) if ENV_SOP_DIR else DEFAULT_DATA_DIR

DISCLAIMER = (
    "This tool provides procedural guidance only. Do not use for clinical decisions or PHI. "
    "Always verify with your site SOPs and Principal Investigator (PI)."
)
FINAL_VERIFICATION_LINE = "Verify with your site SOP and PI before execution."

ROLES = {
    "Clinical Research Coordinator (CRC)": "CRC",
    "Registered Nurse (RN)": "RN",
    "Administrator (Admin)": "ADMIN",
    "Trainee": "TRAINEE",
}

ROLE_SCENARIOS: Dict[str, List[str]] = {
    "CRC": [
        "IP shipment",
        "Missed visit",
        "Adverse event (AE) reporting",
        "Protocol deviation",
        "Monitoring visit preparation",
    ],
    "RN": [
        "Pre-dose checks for IP",
        "AE identification and documentation",
        "Unblinding contingency",
        "Concomitant medication documentation",
    ],
    "ADMIN": [
        "Delegation log management",
        "Regulatory binder maintenance",
        "Safety report distribution",
        "IRB submission packet assembly",
    ],
    "TRAINEE": [
        "SOP basics: GCP overview",
        "Site initiation: required logs",
        "Source documentation fundamentals",
    ],
}

CLARIFYING_QUESTIONS: Dict[str, List[Dict[str, List[str]]]] = {
    "IP shipment": [
        {"Shipment type?": ["Initial shipment", "Resupply", "Return/destruction"]},
        {"Temperature control?": ["Ambient", "Refrigerated (2–8°C)", "Frozen (≤ -20°C)"]},
        {"Chain of custody ready?": ["Yes", "No"]},
    ],
    "Missed visit": [
        {"Visit window status?": ["Within window", "Outside window"]},
        {"Reason documented?": ["Yes", "No"]},
        {"Make-up allowed by protocol?": ["Yes", "No", "Unclear"]},
    ],
    "Adverse event (AE) reporting": [
        {"AE seriousness?": ["Non-serious", "Serious (SAE)"]},
        {"Related to IP?": ["Related", "Not related", "Unknown"]},
        {"Expectedness (per IB)?": ["Expected", "Unexpected", "Unknown"]},
    ],
    "Protocol deviation": [
        {"Deviation type?": ["Minor", "Major"]},
        {"Discovered by?": ["Self-identified", "Monitor", "Sponsor", "IRB", "Other"]},
        {"Subject safety impacted?": ["Yes", "No", "Unknown"]},
    ],
    "Monitoring visit preparation": [
        {"Visit type?": ["SIV", "IMV", "COV"]},
        {"Remote or on-site?": ["Remote", "On-site"]},
        {"Pre-visit docs prepared?": ["Yes", "No"]},
    ],
    "Pre-dose checks for IP": [
        {"Dosing day?": ["Screening", "Baseline", "Treatment day", "Other"]},
        {"Pre-dose labs within window?": ["Yes", "No", "Unknown"]},
        {"Eligibility confirmed?": ["Yes", "No", "Pending PI sign-off"]},
    ],
    "Delegation log management": [
        {"New team member?": ["Yes", "No"]},
        {"Training complete?": ["Yes", "No", "In progress"]},
        {"Signature captured?": ["Wet ink", "eSign", "Not captured"]},
    ],
    "SOP basics: GCP overview": [
        {"Prior experience?": ["None", "<1 year", "1–3 years", "3+ years"]},
    ],
}

OUTPUT_KEYS = ["steps", "required_docs", "escalations", "citations", "compliance", "disclaimer"]

# Optional OpenAI — kept off by default to keep tests deterministic.
USE_OPENAI = False
client = None

@dataclass
class Snippet:
    text: str
    source: str
    score: float
    section_hint: Optional[str] = None  # why: display SOP section when present

@st.cache_data(show_spinner=False)
def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for p in sorted(data_dir.glob("**/*")):
        if p.suffix.lower() == ".txt":
            try:
                docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                pass
        elif p.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(p))
                pages = [page.extract_text() or "" for page in reader.pages]
                docs.append((p.name, "\n".join(pages)))
            except Exception:
                pass
    if not docs:
        docs = [("placeholder.txt", "No SOP files found. Add .txt/.pdf under data/sops.")]
    return docs

@st.cache_data(show_spinner=False)
def build_tfidf_index(docs: List[Tuple[str, str]]):
    sources = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    n_docs = max(1, len(corpus))
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=(0.95 if n_docs > 1 else 1.0))
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, sources, corpus

def retrieve(query: str, tfidf, tfidf_matrix, sources, corpus, k: int = 5,
             openai_embs=None) -> List[Snippet]:
    if not query.strip():
        return []
    q_vec = tfidf.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [Snippet(text=corpus[i][:2000], source=sources[i], score=float(sims[i])) for i in idxs]

def build_query(role_code: str, scenario: str, answers: Dict[str, str]) -> str:
    terms = [role_code, scenario]
    for q, a in answers.items():
        terms.extend([q, a])
    hint = " ".join(terms)
    return f"{scenario} {role_code} {hint} SOP section responsibilities documentation reporting"

def offline_compose(role_label: str, scenario: str, answers: Dict[str, str], snippets: List[Snippet]) -> dict:
    role_short = ROLES.get(role_label, role_label)
    cite_list = sorted({f"Source: {s.source}" for s in snippets})
    steps = [
        f"Confirm {role_short} responsibilities for '{scenario}' using cited SOP sections.",
        "Identify protocol windows/criteria impacted based on clarifying details provided.",
        "Follow site-required documentation order; complete forms/logs referenced in citations.",
        "Record actions with date/time, signer, and cross-references in source records.",
        "Escalate uncertainties to PI/medical lead and document guidance.",
    ]
    required_docs = [
        "Source notes capturing who/what/when/why.",
        "Role-specific log(s) (e.g., delegation, accountability, AE/SAE form).",
        "Correspondence record (e.g., email to sponsor/CRO/IRB) if applicable.",
    ]
    escalations = [
        "Potential subject safety impact.",
        "Regulatory/IRB reporting thresholds met or unclear.",
        "Protocol-required timelines at risk (e.g., SAE 24-hour reporting).",
    ]
    compliance = [
        "Adhere to ICH-GCP E6(R2) and site SOPs.",
        "Use site-approved templates; maintain confidentiality (no PHI in this tool).",
        "Cite SOP section(s) in source; retain chain of custody where relevant.",
    ]
    return {
        "steps": steps,
        "required_docs": required_docs,
        "escalations": escalations,
        "citations": cite_list,
        "compliance": compliance,
        "disclaimer": FINAL_VERIFICATION_LINE,
    }

def to_markdown(role_label: str, scenario: str, answers: Dict[str, str], plan: dict) -> str:
    def bullet(items: List[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) if items else "-"
    md = []
    md.append(f"# CLINI-Q Guidance\n")
    md.append(f"**Role:** {ROLES.get(role_label, role_label)}  \n**Scenario:** {scenario}")
    if answers:
        md.append("**Clarifying Answers:**  " + "; ".join([f"{k}: {v}" for k, v in answers.items()]))
    md.append("\n## Steps")
    md.extend([f"{i+1}. {s}" for i, s in enumerate(plan.get('steps', []))])
    md.append("\n## Required Documentation")
    md.append(bullet(plan.get("required_docs", [])))
    md.append("\n## Escalation Triggers")
    md.append(bullet(plan.get("escalations", [])))
    md.append("\n## SOP Citations")
    md.append(bullet(plan.get("citations", [])))
    md.append("\n## Compliance Reminder")
    md.append(bullet(plan.get("compliance", [])))
    md.append(f"\n> {plan.get('disclaimer', FINAL_VERIFICATION_LINE)}")
    return "\n".join(md)

def to_pdf_bytes(markdown_text: str) -> Optional[bytes]:
    # Fallback-friendly: return None if reportlab not installed.
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.pdfbase.pdfmetrics import stringWidth

        import io
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        width, height = LETTER
        left = 0.75 * inch
        right = 0.75 * inch
        top = height - 0.75 * inch
        line_height = 12
        x = left
        y = top

        for raw_line in markdown_text.split("\n"):
            line = raw_line.replace("\t", "    ")
            max_width = width - left - right
            words = line.split(" ")
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if stringWidth(test, "Helvetica", 10) <= max_width:
                    cur = test
                else:
                    c.setFont("Helvetica", 10)
                    c.drawString(x, y, cur)
                    y -= line_height
                    if y < 0.75 * inch:
                        c.showPage()
                        y = top
                    cur = w
            c.setFont("Helvetica", 10)
            c.drawString(x, y, cur)
            y -= line_height
            if y < 0.75 * inch:
                c.showPage()
                y = top

        c.save()
        pdf = buf.getvalue()
        buf.close()
        return pdf
    except Exception:
        return None

def _render_ui():
    st.set_page_config(page_title="CLINI-Q SOP Navigator", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
          .hero { text-align: left; margin-top: .4rem; }
          .hero h1 { font-size: 1.9rem; font-weight: 800; margin: .2rem 0 .25rem; }
          .hero p  { font-size: 0.95rem; color:#333; max-width: 980px; margin: 0 0 .6rem 0; }
          .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.8rem 1rem; background: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
          <h1>💡 CLINI-Q — Role-based SOP Guidance</h1>
          <p>Role → Scenario → Clarifying Questions → SOP snippets → Structured steps with citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(DISCLAIMER)
    st.divider()

    with st.sidebar:
        st.header("User Setup")
        role_label = st.selectbox("Select your role", list(ROLES.keys()))
        role_code = ROLES[role_label]
        scenario_list = ROLE_SCENARIOS.get(role_code, [])
        scenario = st.selectbox("Select scenario", scenario_list)
        st.subheader("Clarifying questions")
        answers: Dict[str, str] = {}
        for qdef in CLARIFYING_QUESTIONS.get(scenario, []):
            for q, opts in qdef.items():
                answers[q] = st.selectbox(q, opts, key=f"q_{q}")

        st.divider()
        k = st.slider("Evidence snippets", min_value=3, max_value=10, value=5, step=1)
        st.subheader("Data & Keys")
        st.write(f"SOP directory: `{DATA_DIR}`")

    docs = load_documents(DATA_DIR)
    tfidf, tfidf_matrix, sources, corpus = build_tfidf_index(docs)

    query = build_query(ROLES[role_label], scenario, answers)
    st.subheader("Search evidence from SOPs")
    st.write("Query:", query)

    snippets = retrieve(query, tfidf, tfidf_matrix, sources, corpus, k=k)

    if snippets:
        for i, s in enumerate(snippets, 1):
            with st.expander(f"{i}. {s.source}  (relevance {s.score:.2f})", expanded=(i == 1)):
                st.text(s.text if s.text else "(no text)")
    else:
        st.info("No SOP files found. Add .txt or .pdf files under `data/sops`.")

    st.divider()

    col_btn, col_export = st.columns([1, 1])
    with col_btn:
        generate = st.button("Generate CLINI-Q Guidance", type="primary")
    with col_export:
        export_clicked = st.button("Export (PDF / Markdown)")

    if generate:
        plan = offline_compose(role_label, scenario, answers, snippets)

        st.success("Draft guidance generated.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Steps")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for i, step in enumerate(plan.get("steps", []), 1):
                st.markdown(f"**{i}.** {step}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Required Documentation")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for d in plan.get("required_docs", []):
                st.markdown(f"- {d}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("### Escalation Triggers")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for e in plan.get("escalations", []):
                st.markdown(f"- {e}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### SOP Citations")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("; ".join(plan.get("citations", [])) or "-")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Compliance Reminder")
        for item in plan.get("compliance", []):
            st.markdown(f"- {item}")
        st.markdown(f"> {plan.get('disclaimer', FINAL_VERIFICATION_LINE)}")

        st.session_state["last_plan"] = plan
        st.session_state["last_context"] = {
            "role_label": role_label,
            "scenario": scenario,
            "answers": answers,
        }

    if export_clicked:
        last_plan = st.session_state.get("last_plan")
        meta = st.session_state.get("last_context")
        if not last_plan or not meta:
            st.warning("Generate guidance first, then export.")
        else:
            md = to_markdown(meta["role_label"], meta["scenario"], meta["answers"], last_plan)
            pdf_bytes = to_pdf_bytes(md)
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="cliniq_guidance.pdf",
                    mime="application/pdf",
                )
            st.download_button(
                "Download Markdown",
                data=md.encode("utf-8"),
                file_name="cliniq_guidance.md",
                mime="text/markdown",
            )

    st.divider()
    st.caption("MVP scope: No medical advice. No PHI/PII. Not a submission tool. Always verify locally.")

def main():
    _render_ui()

__all__ = [
    "Snippet",
    "load_documents",
    "build_tfidf_index",
    "retrieve",
    "build_query",
    "offline_compose",
    "to_markdown",
    "to_pdf_bytes",
    "DATA_DIR",
    "ROLES",
    "ROLE_SCENARIOS",
    "CLARIFYING_QUESTIONS",
    "FINAL_VERIFICATION_LINE",
]

if __name__ == "__main__":
    main()


# FILE: tests/test_app.py
# Basic tests for CLINI-Q core functions

import os
from pathlib import Path
import json
import types
import importlib

import pytest

# Import app module without running Streamlit UI
@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    # Isolate cache & SOP_DIR per test module
    tmpdir = tmp_path_factory.mktemp("sops")
    os.environ["SOP_DIR"] = str(tmpdir)
    # Streamlit cache writes; isolate by CWD
    cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).parents[1])
        mod = importlib.import_module("app")
        yield mod
    finally:
        os.chdir(cwd)
        os.environ.pop("SOP_DIR", None)

def write_file(dirpath: Path, name: str, text: str):
    p = dirpath / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p

def test_build_query_includes_answers(app_module):
    q = app_module.build_query("CRC", "IP shipment", {"Shipment type?": "Resupply"})
    assert "CRC" in q and "IP shipment" in q and "Resupply" in q

def test_load_documents_placeholder_when_empty(app_module):
    docs = app_module.load_documents(app_module.DATA_DIR)
    assert docs and "No SOP files found" in docs[0][1]

def test_tfidf_retrieval_returns_snippets(app_module, tmp_path):
    # Create sample SOP texts
    sopdir = Path(os.environ["SOP_DIR"])
    write_file(sopdir, "gcp_sop.txt", "AE reporting requires 24-hour notification for SAEs to sponsor.")
    write_file(sopdir, "ip_sop.txt", "IP shipment and chain of custody for CRC. Temperature 2–8°C.")
    # Rebuild index
    docs = app_module.load_documents(sopdir)
    vec, mat, sources, corpus = app_module.build_tfidf_index(docs)
    query = app_module.build_query("CRC", "IP shipment", {"Temperature control?": "Refrigerated (2–8°C)"})
    snips = app_module.retrieve(query, vec, mat, sources, corpus, k=3)
    assert snips and any("ip" in s.source.lower() for s in snips)

def test_offline_compose_schema(app_module):
    snips = [app_module.Snippet(text="foo", source="x.txt", score=0.9)]
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
    # If reportlab installed → bytes, otherwise None. Only validate no exception.
    md = "# Title\nBody"
    out = app_module.to_pdf_bytes(md)
    assert (out is None) or (isinstance(out, (bytes, bytearray)) and len(out) > 0)


# FILE: requirements.txt
streamlit>=1.39.0
scikit-learn>=1.4.0
pypdf>=4.2.0
reportlab>=4.0.9
pytest>=8.0.0


# FILE: README.md
# CLINI-Q (Streamlit) — Tests + Run

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
