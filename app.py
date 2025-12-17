import os
import streamlit as st
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF/Text parsing
from pypdf import PdfReader

# Optional OpenAI usage
USE_OPENAI = False
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

DATA_DIR = Path(__file__).parent / "data" / "sops"

ROLES = {
    "Clinical Research Coordinator (CRC)": "CRC",
    "Registered Nurse (RN)": "RN",
    "Administrator (Admin)": "ADMIN",
    "Trainee": "TRAINEE",
}

DEFAULT_SCENARIOS = [
    "What steps are required before dosing an investigational product (IP)?",
    "How do I document an adverse event (AE) for a subject?",
    "What is the procedure to report a protocol deviation?",
    "How do I manage delegation logs for new team members?",
]

DISCLAIMER = (
    "This tool provides procedural guidance only. Verify against your site SOPs and Principal "
    "Investigator (PI) instructions before execution. Do not use this tool for clinical decisions "
    "or to handle protected health information."
)

# ---------------- Data classes & helpers ----------------
@dataclass
class Snippet:
    text: str
    source: str
    score: float

@st.cache_data(show_spinner=False)
def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (source, text) tuples from .txt and .pdf files."""
    docs = []
    for p in sorted(data_dir.glob("**/*")):
        if p.suffix.lower() == ".txt":
            try:
                docs.append((str(p.name), p.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                pass
        elif p.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(p))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages)
                docs.append((str(p.name), text))
            except Exception:
                pass
    return docs

@st.cache_data(show_spinner=False)
def build_index(docs: List[Tuple[str, str]]):
    """Build and cache a TF-IDF index."""
    sources = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    if not corpus:
        corpus = ["placeholder text for empty corpus"]
        sources = ["placeholder.txt"]
    # Safe params for tiny corpora to avoid max_df/min_df conflicts
    n_docs = len(corpus)
    if n_docs < 2:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=1.0)
    else:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, sources, corpus

def retrieve(query: str, vectorizer, matrix, sources, corpus, k: int = 5) -> List[Snippet]:
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    idxs = sims.argsort()[::-1][:k]
    return [Snippet(text=corpus[i][:2000], source=sources[i], score=float(sims[i])) for i in idxs]

def draft_prompt(role: str, scenario: str, snippets: List[Snippet]) -> str:
    context_blocks = "\n\n".join([f"[Source: {s.source}]\n{s.text}" for s in snippets])
    role_short = ROLES.get(role, role)
    prompt = f"""You are CLINI-Q, an intelligent SOP navigator for clinical research operations.
User role: {role_short}
User scenario/question: {scenario}

Context (SOP snippets):
{context_blocks}

Write a step-by-step, role-specific procedural guidance (numbered steps). For each key step, cite the relevant [Source: …].
Include a short 'Compliance Reminders' list after the steps.
Never give medical advice. Always end with: 'Verify against site SOP and PI instructions before execution.'
Return JSON with keys: steps (list of strings), citations (list of 'Source: file.pdf' strings), compliance (list of strings), disclaimer (string)."""
    return prompt

def offline_plan(role: str, scenario: str, snippets: List[Snippet]) -> dict:
    steps = [
        f"Confirm scope for {role} and locate applicable SOP section(s).",
        "Review inclusion/exclusion criteria and protocol requirements relevant to the scenario.",
        "Follow site-required documentation sequence (forms/logs) as referenced in cited sources.",
        "Record actions with date/time, signer, and cross-references in the source record.",
        "Escalate uncertainties to PI/clinical lead; document clarifications."
    ]
    citations = sorted({f"Source: {s.source}" for s in snippets})
    compliance = [
        "Adhere to ICH-GCP E6(R2) principles.",
        "Use site-approved templates and logs.",
        "Do not enter PHI into this tool; maintain confidentiality.",
    ]
    return {
        "steps": steps,
        "citations": list(citations),
        "compliance": compliance,
        "disclaimer": "Verify against site SOP and PI instructions before execution.",
    }

def generate_guidance(role: str, scenario: str, snippets: List[Snippet]) -> dict:
    if USE_OPENAI and client:
        prompt = draft_prompt(role, scenario, snippets)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful, compliance-focused assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            import json
            try:
                data = json.loads(content)
            except Exception:
                data = {"steps": [content], "citations": [s.source for s in snippets], "compliance": [],
                        "disclaimer": "Verify against site SOP and PI instructions before execution."}
            return data
        except Exception as e:
            st.warning(f"OpenAI call failed; running in offline mode. ({e})")
            return offline_plan(role, scenario, snippets)
    else:
        return offline_plan(role, scenario, snippets)

# ---------------- UI ----------------
st.set_page_config(page_title="CLINI-Q SOP Navigator", page_icon="🧭", layout="wide")

# Global styles to mimic RISe layout
st.markdown(
    """
    <style>
      .hero { text-align: center; margin-top: .3rem; }
      .hero h1 { font-size: 2.2rem; font-weight: 800; margin: .2rem 0 .2rem; }
      .hero h2 { font-size: 1.1rem; font-weight: 700; font-style: italic; margin: .1rem 0 .6rem; }
      .hero p  { font-size: 1rem; color:#333; max-width: 950px; margin: 0 auto .8rem; }
      .divider-strong { border-top: 4px solid #222; margin: .2rem 0 1.2rem; }
      .label { color:#6b7280; font-size:.9rem; margin:.2rem 0 .2rem; }
      .hint  { color:#6b7280; font-size:.85rem; }
      .chiprow { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.4rem; }
      .chiprow button { border-radius:9999px; padding:.35rem .8rem; border:1px solid #e5e7eb; background:#f9fafb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Centered big logo ---
center = st.columns([1, 2, 1])[1]
with center:
    st.image(str(Path(__file__).parent / "assets" / "cliniq_logo.png"), use_column_width=True)

# --- Title, subtitle, description (exactly like RISe pattern) ---
st.markdown(
    """
    <div class="hero">
      <h1>🛡️ CLINI-Q Clinical Trial SOP Assistant 🛡️</h1>
      <h2>💡 Smart Assistant for Clinical Trial SOP Navigation</h2>
      <p>
        I am trained on institutional Standard Operating Procedures (SOPs) and compliance frameworks,
        CLINI-Q helps research teams navigate essential documentation, regulatory requirements, and
        Good Clinical Practice (GCP) standards with clarity and confidence.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="divider-strong"></div>', unsafe_allow_html=True)

# Disclaimer just below like the example
st.caption(DISCLAIMER)

# ---------- Primary input block (RISe-style) ----------
st.markdown('<div class="label">📎 Upload a file for reference (optional)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here", type=["pdf", "docx", "txt"], label_visibility="collapsed")
st.markdown('<div class="hint">Limit 200MB per file • PDF, DOCX, TXT</div>', unsafe_allow_html=True)

# categories inferred from filenames; fallback to "All Categories"
docs = load_documents(DATA_DIR)
vectorizer, matrix, sources, corpus = build_index(docs)
categories = ["All Categories"] + sorted({Path(s).stem.split("_")[0].title() for s in sources})
st.markdown('<div class="label">📁 Select a category:</div>', unsafe_allow_html=True)
category = st.selectbox("", options=categories, label_visibility="collapsed")

st.markdown('<div class="label">💬 What would you like me to help you with?</div>', unsafe_allow_html=True)
user_text = st.text_area("", placeholder="Type your question...", height=80, label_visibility="collapsed")

st.markdown('<div class="label">💡 Try asking one of these:</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
suggestions = DEFAULT_SCENARIOS
with c1:
    if st.button(suggestions[0]): user_text = suggestions[0]
with c2:
    if st.button(suggestions[1]): user_text = suggestions[1]
with c3:
    if st.button(suggestions[2]): user_text = suggestions[2]
with c4:
    if st.button(suggestions[3]): user_text = suggestions[3]

st.divider()

# ---------- Guidance + evidence ----------
left, right = st.columns([1, 2])

with left:
    st.header("Setup")
    role = st.selectbox("Your role", list(ROLES.keys()), index=0)
    k = st.slider("Evidence snippets", min_value=3, max_value=10, value=5, step=1)
    go = st.button("Generate CLINI-Q Guidance", type="primary")

with right:
    if user_text:
        st.subheader("Query")
        st.write(user_text)

if go and user_text:
    # Optionally do something with uploaded file (not indexed into corpus in this MVP)
    snippets = retrieve(user_text, vectorizer, matrix, sources, corpus, k=k)
    plan = generate_guidance(role, user_text, snippets)
    st.success("Draft guidance generated.")

    st.markdown("### Step-by-step guidance")
    for i, step in enumerate(plan.get("steps", []), 1):
        st.markdown(f"**{i}.** {step}")

    if plan.get("citations"):
        st.markdown("### Citations")
        st.write("; ".join(plan["citations"]))

    if plan.get("compliance"):
        st.markdown("### Compliance Reminders")
        for item in plan["compliance"]:
            st.markdown(f"- {item}")

    st.markdown(f"> {plan.get('disclaimer', '')}")

st.divider()
st.caption("MVP scope: No medical advice. No PHI/PII. Not a submission tool. Always verify locally.")

