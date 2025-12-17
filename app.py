#!/usr/bin/env python3
# CLINI-Q: Intelligent SOP Navigator (Streamlit MVP)
# Author: Software Architect GPT
# License: MIT
#
# Summary:
# - Lets users choose a role (CRC, RN, Admin, Trainee) and describe a scenario.
# - Retrieves the most relevant SOP snippets from local /data/sops using TF-IDF.
# - Optionally calls OpenAI (if OPENAI_API_KEY is set in Streamlit secrets) to draft
#   structured, role-specific procedural guidance with citations to the retrieved SOPs.
# - Falls back to a simple, offline "rule-based summary" if no key is provided.
#
# Anti-scope guardrails: The app does NOT provide medical advice, does NOT
# access PHI/PII, and is NOT an automation for IRB/Regulatory submissions.
# It provides procedural guidance only and reminds users to verify with site SOP & PI.
#
# Deployment: Works on Streamlit Community Cloud. Put OPENAI_API_KEY in Streamlit secrets.
#
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
    # Prefer Streamlit secrets when running on Streamlit Cloud
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
    "How do I document an adverse event (AE) for a subject?",
    "What steps are required before dosing an investigational product (IP)?",
    "How do I manage delegation logs for new team members?",
    "What is the procedure to report a protocol deviation?",
]

DISCLAIMER = (
    "This tool provides procedural guidance only. Verify against your site SOPs and Principal "
    "Investigator (PI) instructions before execution. Do not use this tool for clinical decisions "
    "or to handle protected health information."
)

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
        # keep vectorizer minimally fit to avoid errors
        corpus = ["placeholder text for empty corpus"]
        sources = ["placeholder.txt"]
    # Avoid max_df/min_df conflict when there are very few documents
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
    results = []
    for i in idxs:
        results.append(Snippet(text=corpus[i][:2000], source=sources[i], score=float(sims[i])))
    return results

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
    # Deterministic fallback without LLM
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

# --- Hero header with logo and description ---
with st.container():
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        st.image(str(Path(__file__).parent / "assets" / "cliniq_logo.png"), use_column_width=True)
    with col2:
        st.markdown("# CLINI-Q — Smart Assistant for Clinical Trial SOP Navigation")
        st.markdown(
            "I am trained on institutional Standard Operating Procedures (SOPs) and compliance frameworks, "
            "CLINI-Q helps research teams navigate essential documentation, regulatory requirements, and "
            "Good Clinical Practice (GCP) standards with clarity and confidence."
        )
st.divider()

st.caption(DISCLAIMER)

with st.sidebar:
    st.header("User Setup")
    role = st.selectbox("Your role", list(ROLES.keys()))
    scenario = st.selectbox("Common scenarios", DEFAULT_SCENARIOS)
    custom = st.text_area("…or describe your scenario", placeholder="Describe your procedural question…", height=100)
    query = custom.strip() or scenario
    k = st.slider("Evidence snippets", min_value=3, max_value=10, value=5, step=1)
    st.divider()
    st.subheader("Data & Keys")
    st.write("Place SOP files (.txt/.pdf) in `data/sops`. On Streamlit Cloud, upload via repo.")
    st.write("Set `OPENAI_API_KEY` in Streamlit Secrets for LLM drafting (optional).")

docs = load_documents(DATA_DIR)
vectorizer, matrix, sources, corpus = build_index(docs)

st.subheader("Search evidence from SOPs")
st.write("Query:", query)
snippets = retrieve(query, vectorizer, matrix, sources, corpus, k=k)

if snippets:
    for i, s in enumerate(snippets, 1):
        with st.expander(f"{i}. {s.source}  (relevance {s.score:.2f})", expanded=(i==1)):
            st.text(s.text[:2000] if s.text else "(no text)")
else:
    st.info("No SOP files found. Add .txt or .pdf files under `data/sops`.")

st.divider()
if st.button("Generate CLINI-Q Guidance", type="primary"):
    plan = generate_guidance(role, query, snippets)
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
else:
    st.info("Adjust your scenario and click **Generate CLINI-Q Guidance**.")

st.divider()
st.caption("MVP scope: No medical advice. No PHI/PII. Not a submission tool. Always verify locally.")
