import os
import base64
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
from PIL import Image
from difflib import SequenceMatcher, get_close_matches

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------
APP_TITLE = "CLINI-Q • SOP Navigator"
ASSETS_DIR = Path(__file__).parent / "assets"
ICON_PATH = ASSETS_DIR / "icon.png"                  # <-- put your icon here (used everywhere)
LOGO_PATH = ASSETS_DIR / "cliniq_logo.png"           # optional wide header logo (falls back to icon)
FAQ_CSV   = Path(__file__).parent / "cliniq_faq.csv" # optional: Category,Question,Answer

DEFAULT_SOP_DIR = Path(__file__).parent / "data" / "sops"
DATA_DIR = Path(os.environ.get("SOP_DIR", "").strip() or DEFAULT_SOP_DIR)

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

# ------------------ TYPES ------------------
@dataclass
class Snippet:
    text: str
    source: str
    score: float

# ------------------ IMAGE/STYLE HELPERS ------------------
def _img_to_b64(path: Path) -> str:
    try:
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

def _load_page_icon():
    # Use custom icon if present; fallback to emoji to avoid runtime errors
    try:
        if ICON_PATH.exists():
            return Image.open(ICON_PATH)
    except Exception:
        pass
    return "🧭"

def _show_bubble(html: str, avatar_b64: str):
    st.markdown(
        f"""
        <div style='display:flex;align-items:flex-start;margin:10px 0;'>
            <img src='data:image/png;base64,{avatar_b64}' width='40' style='margin-right:10px;border-radius:8px;'/>
            <div style='background:#f6f6f6;padding:12px;border-radius:120px;max-width:75%;'>
                {html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------ KNOWLEDGE ------------------
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
def build_index(docs: List[Tuple[str, str]]):
    sources = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    n = len(corpus)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=(0.95 if n > 1 else 1.0))
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, sources, corpus

def retrieve(query: str, vectorizer, matrix, sources, corpus, k: int = 5) -> List[Snippet]:
    if not query.strip():
        return []
    sims = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [Snippet(text=corpus[i][:2000], source=sources[i], score=float(sims[i])) for i in idxs]

def build_query(role_code: str, scenario: str, answers: Dict[str, str]) -> str:
    terms = [role_code, scenario] + [t for kv in answers.items() for t in kv]
    hint = " ".join(terms)
    return f"{scenario} {role_code} {hint} SOP section responsibilities documentation reporting"

def compose_guidance(role_label: str, scenario: str, answers: Dict[str, str], snippets: List[Snippet]) -> dict:
    role_short = ROLES.get(role_label, role_label)
    cites = sorted({f"Source: {s.source}" for s in snippets})
    steps = [
        f"Confirm {role_short} responsibilities for '{scenario}' using cited SOP sections.",
        "Identify protocol windows/criteria impacted based on clarifying details provided.",
        "Follow site-required documentation order; complete forms/logs referenced in citations.",
        "Record actions with date/time, signer, and cross-references in source records.",
        "Escalate uncertainties to PI/medical lead and document guidance.",
    ]
    docs = [
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
        "required_docs": docs,
        "escalations": escalations,
        "citations": cites,
        "compliance": compliance,
        "disclaimer": FINAL_VERIFICATION_LINE,
    }

# ------------------ APP (MSU-style flow with ICON) ------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=_load_page_icon(), layout="wide")

    icon_b64 = _img_to_b64(ICON_PATH)  # used for header + chat bubble
    logo_b64 = _img_to_b64(LOGO_PATH) or icon_b64

    # Header with icon (MSU-style left alignment)
    st.markdown(
        """
        <style>
          .hero { text-align:left; margin-top:.3rem; }
          .hero h1 { font-size:2.05rem; font-weight:800; margin:-1; }
          .hero p  { font-size:1rem; color:#333; max-width:980px; margin:.25rem 0 0 0; }
          .divider-strong { border-top:4px solid #222; margin:.4rem 0 1.0rem; }
          .card { border:1px solid #e5e7eb; border-radius:12px; padding:.8rem 1rem; background:#fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="hero-wrap">
            <img src="data:image/png;base64,{(logo_b64 or icon_b64)}" style="height:164px;border-radius:8px;"/>
            <div class="hero">
                <h1>💡Smart Assistant for Clinical Trial SOP Navigation</h1>
                <p>I am trained on institutional Standard Operating Procedures (SOPs) and compliance frameworks, helping research teams navigate essential documentation, regulatory requirements, and Good Clinical Practice (GCP) standards with clarity and confidence.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider-strong"></div>', unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    # Upload hint (visual parity with MSU)
    uploaded = st.file_uploader("📎 Upload a reference file (optional)", type=["pdf", "docx", "txt"])
    if uploaded:
        st.success(f"Uploaded file: {uploaded.name}")

    # Session state
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("suggested", [])
    st.session_state.setdefault("last_category", "")
    st.session_state.setdefault("clear_input", False)

    # Sidebar: Category + Role + Scenario + Clarifiers
    with st.sidebar:
        st.header("User Setup")

        try:
            faq_df = pd.read_csv(FAQ_CSV).fillna("")
            categories = ["All Categories"] + sorted(faq_df["Category"].unique().tolist())
        except Exception:
            faq_df = pd.DataFrame(columns=["Category", "Question", "Answer"])
            categories = ["All Categories"]

        category = st.selectbox("📂 Knowledge category (optional)", categories)

        role_label = st.selectbox("🎭 Your role", list(ROLES.keys()))
        role_code = ROLES[role_label]
        scenario = st.selectbox("📌 Scenario", ROLE_SCENARIOS.get(role_code, []))

        st.subheader("Clarifying questions")
        answers: Dict[str, str] = {}
        for qdef in CLARIFYING_QUESTIONS.get(scenario, []):
            for q, opts in qdef.items():
                answers[q] = st.selectbox(q, opts, key=f"q_{q}")

        k = st.slider("Evidence snippets", min_value=3, max_value=10, value=5, step=1)
        st.divider()
        st.subheader("Data & Keys")
        st.write(f"SOP directory: `{DATA_DIR}`")
        st.write("Optional CSV: `cliniq_faq.csv` (Category, Question, Answer).")

    # Reset chat on category change
    if st.session_state["last_category"] != category:
        st.session_state["chat"] = []
        st.session_state["suggested"] = []
        st.session_state["last_category"] = category

    # Chat input + suggestions
    question = st.text_input(
        "💬 What would you like me to help you with?",
        value="" if st.session_state["clear_input"] else "",
        placeholder="Ask about steps, documentation, reporting timelines…",
    )
    st.session_state["clear_input"] = False

    selected_df = (
        faq_df if faq_df.empty or category == "All Categories"
        else faq_df[faq_df["Category"] == category]
    )

    if not question.strip():
        st.markdown("💬 Try asking one of these:")
        examples = (
            selected_df["Question"].head(3).tolist()
            if not selected_df.empty
            else [f"What are the steps for {s}?" for s in ROLE_SCENARIOS.get(role_code, [])[:3]]
        )
        cols = st.columns(len(examples)) if examples else []
        for i, q in enumerate(examples):
            if st.button(q, key=f"ex_{i}"):
                st.session_state["chat"].append({"role": "user", "content": q})
                if not selected_df.empty and q in selected_df["Question"].values:
                    ans = selected_df[selected_df["Question"] == q].iloc[0]["Answer"]
                    st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {ans}"})
                st.session_state["clear_input"] = True
                st.rerun()

    # Show chat with icon avatar
    st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
    for msg in st.session_state["chat"]:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align:right;margin:10px 0;'>
                    <div style='display:inline-block;background:#e6f7ff;padding:12px;border-radius:12px;max-width:75%;'>
                        <b>You:</b> {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _show_bubble(msg["content"], icon_b64 or "")

    # Autocomplete suggestions on typing
    if question.strip() and not selected_df.empty:
        matches = [q for q in selected_df["Question"].tolist() if question.lower() in q.lower()][:5]
        if matches:
            st.markdown("<div style='margin-top:5px;'><b>Suggestions:</b></div>", unsafe_allow_html=True)
            for s in matches:
                if st.button(s, key=f"suggest_{s}"):
                    st.session_state["chat"].append({"role": "user", "content": s})
                    ans = selected_df[selected_df["Question"] == s].iloc[0]["Answer"]
                    st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {ans}"})
                    st.session_state["clear_input"] = True
                    st.rerun()

    # Submit → exact/close match from CSV (MSU logic)
    if st.button("Submit") and question.strip():
        st.session_state["chat"].append({"role": "user", "content": question})
        prev_suggestions = st.session_state["suggested"]
        st.session_state["suggested"] = []
        st.session_state["clear_input"] = True

        if not selected_df.empty:
            all_q = selected_df["Question"].tolist()
            best, score = None, 0.0
            for q in all_q:
                s = SequenceMatcher(None, question.lower(), q.lower()).ratio()
                if s > score:
                    best, score = q, s
            if best and score >= 0.85:
                row = selected_df[selected_df["Question"] == best].iloc[0]
                html = f"<b>Answer:</b> {row['Answer']}<br><i>(Category: {row['Category']})</i>"
                st.session_state["chat"].append({"role": "assistant", "content": html})
            else:
                if prev_suggestions:
                    sq = prev_suggestions[0]
                    row = faq_df[faq_df["Question"] == sq].iloc[0]
                    html = f"<b>Answer:</b> {row['Answer']}<br><i>(Category: {row['Category']})</i>"
                    st.session_state["chat"].append({"role": "assistant", "content": html})
                else:
                    top = get_close_matches(question, faq_df["Question"].tolist(), n=3, cutoff=0.4)
                    if top:
                        guessed_cat = faq_df[faq_df["Question"] == top[0]].iloc[0]["Category"]
                        html = (
                            f"I couldn't find an exact match, but your question seems related to <b>{guessed_cat}</b>.<br><br>"
                            "Here are similar questions:<br>" +
                            "".join(f"{i}. {q}<br>" for i, q in enumerate(top, start=1)) +
                            "<br>Select one below to see its answer."
                        )
                        st.session_state["chat"].append({"role": "assistant", "content": html})
                        st.session_state["suggested"] = top
                    else:
                        st.session_state["chat"].append({"role": "assistant", "content": "I couldn't find a close match. Please try rephrasing."})
        else:
            st.session_state["chat"].append({"role": "assistant", "content": "Thanks—see SOP-based guidance below."})

        st.rerun()

    # Buttons for suggested similar questions
    if st.session_state["suggested"]:
        st.markdown("<div style='margin-top:15px;'><b>Choose a question:</b></div>", unsafe_allow_html=True)
        for i, q in enumerate(st.session_state["suggested"]):
            if st.button(q, key=f"choice_{i}"):
                row = faq_df[faq_df["Question"] == q].iloc[0]
                st.session_state["chat"].append({"role": "assistant", "content": f"<b>Answer:</b> {row['Answer']}"})
                st.session_state["suggested"] = []
                st.session_state["clear_input"] = True
                st.rerun()

    # ----- SOP Retrieval & Guidance -----
    st.divider()
    docs = load_documents(DATA_DIR)
    vectorizer, matrix, sources, corpus = build_index(docs)

    sop_query = build_query(ROLES[role_label], scenario, answers)
    st.subheader("🔎 Search evidence from SOPs")
    st.write("Query:", sop_query)

    snippets = retrieve(sop_query, vectorizer, matrix, sources, corpus, k=k)
    if snippets:
        for i, s in enumerate(snippets, 1):
            with st.expander(f"{i}. {s.source}  (relevance {s.score:.2f})", expanded=(i == 1)):
                st.text(s.text if s.text else "(no text)")
    else:
        st.info("No SOP files found. Add .txt or .pdf files under `data/sops`.")

    st.divider()
    if st.button("Generate CLINI-Q Guidance", type="primary"):
        plan = compose_guidance(role_label, scenario, answers, snippets)

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
    else:
        st.info("Adjust your inputs and click **Generate CLINI-Q Guidance**.")

    # Download chat history (unchanged, visible under chat)
    if st.session_state["chat"]:
        chat_text = ""
        for m in st.session_state["chat"]:
            who = "You" if m["role"] == "user" else "Assistant"
            chat_text += f"{who}: {m['content']}\n\n"
        b64 = base64.b64encode(chat_text.encode()).decode()
        st.markdown(
            f'<a href="data:file/txt;base64,{b64}" download="cliniq_chat_history.txt">📥 Download Chat History</a>',
            unsafe_allow_html=True,
        )

    st.caption("© 2025 CLINIQ ⚖️Disclaimer: This is a demo tool only. For official guidanceNo PHI/PII without accessing sensitive data. For official guidance, refer to your office policies.")

# -------- import-safe entrypoint --------
if __name__ == "__main__":
    main()
