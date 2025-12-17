import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "CLINI-Q • Role-based SOP Guidance"

# Let tests/ops override SOP folder via env var
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
    "CRC": ["IP shipment", "Missed visit", "Adverse event (AE) reporting", "Protocol deviation", "Monitoring visit preparation"],
    "RN": ["Pre-dose checks for IP", "AE identification and documentation", "Unblinding contingency", "Concomitant medication documentation"],
    "ADMIN": ["Delegation log management", "Regulatory binder maintenance", "Safety report distribution", "IRB submission packet assembly"],
    "TRAINEE": ["SOP basics: GCP overview", "Site initiation: required logs", "Source documentation fundamentals"],
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

@dataclass
class Snippet:
    text: str
    source: str
    score: float
    section_hint: Optional[str] = None  # why: show SOP section label if known

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

def retrieve(query: str, tfidf, tfidf_matrix, sources, corpus, k: int = 5) -> List[Snippet]:
    if not query.strip():
        return []
    sims = cosine_similarity(tfidf.transform([query]), tfidf_matrix).ravel()
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
        st.subheader("Data")
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
                st.download_button("Download PDF", data=pdf_bytes, file_name="cliniq_guidance.pdf", mime="application/pdf")
            st.download_button("Download Markdown", data=md.encode("utf-8"), file_name="cliniq_guidance.md", mime="text/markdown")

    st.divider()
    st.caption("MVP scope: No medical advice. No PHI/PII. Always verify locally.")

def main():
    _render_ui()

if __name__ == "__main__":
    main()
