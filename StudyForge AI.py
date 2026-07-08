"""
StudyForge AI — Multi-Agent Exam Prep Assistant. Streamlit production app.

Architecture: 1 Orchestrator + 5 specialized agents, free-tier OpenRouter models only.
API key is backend-only by design: read from .env / .streamlit/secrets.toml. There is
no UI input path for it anywhere in this file.

    Orchestrator Agent (LLM)  -> plans focus areas + per-agent instructions
        |
        v
    1. Concept Explainer Agent  -> explains the topic with a worked example
        |
        v
    2. Question Generator Agent -> writes practice questions across difficulty tiers
        |
        v
    3. Solution Agent           -> full worked solutions / answer key
        |
        v
    4. Weak-Area Diagnostic Agent -> flags common pitfalls for this topic
        |
        v
    5. Study Plan Agent         -> day-by-day revision schedule to the exam date

Pipeline shape: concept -> questions -> solutions is one branch, weak_areas ->
study_plan is a second, independent branch (it only needs the topic/confidence,
not the generated questions). Both branches fan out from the orchestrator and
run concurrently, which is roughly half the wall-clock time of running all six
agents strictly in series.

Every LLM-backed agent has its own ordered fallback chain of free models, so a single
model returning 404 (retired) or 429 (rate-limited) doesn't sink the whole run.

Setup:
    pip install langgraph langchain-core requests streamlit python-dotenv pypdf

    API key: put it in a .env file next to this script:
        OPENROUTER_API_KEY=sk-or-v1-...
    or in .streamlit/secrets.toml:
        OPENROUTER_API_KEY = "sk-or-v1-..."

    Get a key at https://openrouter.ai/keys

Run:
    streamlit run study_prep_agent.py
"""

import os
import re
import json
import time
import logging
import threading
from datetime import datetime, date
from typing import TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("study_prep_agent")

# ============================================================
# CONFIG
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free-model fallback chains, one per agent, verified live against
# openrouter.ai/collections/free-models. As of this build, NEITHER Qwen NOR DeepSeek
# have any free-tier listing on OpenRouter at all - their :free variants were pulled
# platform-wide - so chains built around those IDs would silently 404 straight through
# to the openrouter/free safety net every time. These chains lead with the smallest /
# lowest-latency models actually confirmed free right now, add a distinct-provider
# model as a third step for resilience (so one provider having a bad day doesn't take
# out every chain at once), and fall back to slightly larger ones for the two agents
# (concept, solutions) where answer quality matters most:
#   - nvidia/nemotron-nano-9b-v2:free        - 9B, smallest free model available
#   - google/gemma-4-26b-a4b-it:free         - MoE, only 3.8B active params/token
#   - openai/gpt-oss-20b:free                - 21B, 3.6B active, tuned for low latency
#   - nvidia/nemotron-3-nano-30b-a3b:free    - 30B total, sparse MoE
#   - z-ai/glm-4.5-air:free                  - different provider entirely (Zhipu AI),
#                                              inserted before the last-resort router
#                                              so an NVIDIA/Google/OpenAI outage isn't
#                                              a single point of failure across agents
# The free-model roster still rotates - spot-check openrouter.ai/models?max_price=0
# occasionally and swap any ID below that starts 404'ing.
AGENT_MODEL_FALLBACKS = {
    "orchestrator": [
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-4-26b-a4b-it:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
    "concept": [
        "google/gemma-4-26b-a4b-it:free",
        "openai/gpt-oss-20b:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
    "questions": [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "google/gemma-4-26b-a4b-it:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
    "solutions": [
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
    "weak_areas": [
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-4-26b-a4b-it:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
    "study_plan": [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "z-ai/glm-4.5-air:free",
        "openrouter/free",
    ],
}

# Minimum spacing (seconds) between actual outbound requests, shared across ALL
# agents/threads, to stay under OpenRouter's 20 req/min free-tier cap (3s * 20 = 60s).
# Unlike a flat "sleep 3s before every call", this only waits the remaining time
# needed since the last request actually went out - if a previous call's own network
# latency already ate up 3+ seconds, the next call fires immediately.
MIN_CALL_INTERVAL_SECONDS = 3.0

_rate_lock = threading.Lock()
_last_call_at = {"t": 0.0}


def _throttle():
    """Blocks just long enough to keep global request spacing >= MIN_CALL_INTERVAL_SECONDS.
    Thread-safe so the two parallel pipeline branches share one rate budget instead of
    each independently sleeping and together blowing through the 20 req/min cap.
    """
    with _rate_lock:
        now = time.monotonic()
        wait = MIN_CALL_INTERVAL_SECONDS - (now - _last_call_at["t"])
        if wait > 0:
            time.sleep(wait)
        _last_call_at["t"] = time.monotonic()


OPENROUTER_API_KEY = "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"


def resolve_api_key() -> str:
    """Returns the OpenRouter API key hardcoded above."""
    return OPENROUTER_API_KEY


def call_llm(agent_name: str, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Single entry point for all LLM calls. Walks the agent's free-model fallback
    chain in order, moving on immediately on 404 and after a short backoff on 429,
    until one model responds or the chain is exhausted.
    """
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key configured. Add one via .streamlit/secrets.toml or a .env file."
        )

    chain = AGENT_MODEL_FALLBACKS.get(agent_name)
    if not chain:
        raise ValueError(f"No model chain configured for agent '{agent_name}'")

    def _post(model_id: str) -> str:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    last_error = None
    for i, model_id in enumerate(chain):
        _throttle()
        try:
            return _post(model_id)
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            last_error = e
            remaining = chain[i + 1:]
            if not remaining:
                break
            if status == 429:
                log.warning(f"'{model_id}' rate-limited (429), backing off then trying '{remaining[0]}'")
                time.sleep(MIN_CALL_INTERVAL_SECONDS)
            else:
                log.warning(f"'{model_id}' failed ({e}), trying '{remaining[0]}'")

    raise RuntimeError(
        f"All free models for agent '{agent_name}' failed. Last error: {last_error}. "
        "Free-tier daily quota may be exhausted, or the free model roster has rotated - "
        "check openrouter.ai/models?max_price=0."
    ) from last_error


def _extract_json(raw: str, default):
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(f"Could not parse JSON, using default. Raw: {raw[:300]}")
        return default


# ============================================================
# PDF INGESTION
# ============================================================

MAX_SOURCE_MATERIAL_CHARS = 15000  # cap so one huge PDF doesn't blow every agent's token budget
MAX_ORCHESTRATOR_EXCERPT_CHARS = 3000  # orchestrator only needs enough to infer topic/focus


def extract_pdf_text(uploaded_file, max_chars: int = MAX_SOURCE_MATERIAL_CHARS) -> str:
    """Extracts text from an uploaded PDF (Streamlit UploadedFile), page by page,
    and caps the result at max_chars. Raises ValueError with a user-facing message
    if the PDF can't be opened or has no extractable text (e.g. a scanned image PDF
    with no OCR layer).
    """
    try:
        reader = PdfReader(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read this PDF: {e}") from e

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            continue

    full_text = "\n\n".join(t for t in pages_text if t.strip()).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found in this PDF. It may be scanned/image-based "
            "without an OCR text layer — try a text-based PDF instead."
        )

    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[...truncated for length...]"
    return full_text


def _material_block(state: "StudyState", max_chars: int = MAX_SOURCE_MATERIAL_CHARS) -> str:
    """Formats the uploaded source material (if any) as a prompt block, trimmed to
    max_chars so per-agent prompts stay within a reasonable token budget even though
    the full text is already capped once at extraction time.
    """
    material = (state.get("source_material") or "").strip()
    if not material:
        return ""
    if len(material) > max_chars:
        material = material[:max_chars] + "\n\n[...truncated for this prompt...]"
    return f"\n\nSource material (uploaded by the student, ground all content in this where relevant):\n{material}"


# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrator_agent(state: "StudyState") -> dict:
    """LLM-backed orchestrator. Reads the topic, confidence level, and days until
    the exam, and produces a short task plan threaded into every downstream agent's
    prompt. Falls back to an empty plan (never kills the run) if all models fail.

    If the student uploaded a PDF instead of typing a topic, this is also the agent
    that infers the actual topic from the source material excerpt.

    Returns only the keys this node actually sets. LangGraph's default reducer
    ("last value wins") rejects two nodes writing the same key in one superstep -
    since concept_agent and weak_area_agent run concurrently after this node, every
    agent below returns a small partial-update dict instead of the whole mutated
    state, so the two branches never collide on unrelated keys like 'topic'.
    """
    needs_topic = not (state.get("topic") or "").strip() and state.get("source_material")

    system_prompt = (
        "You are the orchestrator agent for a 5-agent exam-prep pipeline: concept, "
        "questions, solutions, weak_areas, study_plan. Given a study topic (or, if "
        "no topic was given, an excerpt of uploaded source material to infer one from), "
        "a self-rated confidence (1-5), and days remaining until the exam, produce a "
        "short task plan. "
        'Respond ONLY with JSON: {"topic": "...", "focus_areas": ["...", "..."], '
        '"instructions": {"concept": "...", "questions": "...", "solutions": "...", '
        '"weak_areas": "...", "study_plan": "..."}}. '
        'Set "topic" to a short topic name: either restate the given topic verbatim, '
        "or, if none was given, infer the single most central topic/subject of the "
        "source material excerpt. "
        "Each instruction is one concise sentence telling that agent what to prioritize "
        "given the confidence level and urgency."
    )
    user_prompt = (
        f"Topic: {state.get('topic') or '(none given — infer from source material below)'}\n"
        f"Self-rated confidence (1=none, 5=strong): {state.get('confidence', 3)}\n"
        f"Days until exam: {state.get('days_until_exam', 'not specified')}"
        f"{_material_block(state, max_chars=MAX_ORCHESTRATOR_EXCERPT_CHARS)}"
    )
    try:
        raw = call_llm("orchestrator", system_prompt, user_prompt)
        plan = _extract_json(raw, {"focus_areas": [], "instructions": {}})
    except Exception as e:
        log.warning(f"Orchestrator planning failed, proceeding with an empty plan: {e}")
        plan = {"focus_areas": [], "instructions": {}}

    updates = {"orchestrator_plan": plan}
    if needs_topic:
        inferred = (plan.get("topic") or "").strip()
        updates["topic"] = inferred if inferred else "Uploaded document"
    if not state.get("run_started_at"):
        updates["run_started_at"] = datetime.now().isoformat()
    return updates


def _plan_instruction(state: "StudyState", agent_name: str) -> str:
    instr = (state.get("orchestrator_plan") or {}).get("instructions", {}).get(agent_name, "")
    return f"\n\nOrchestrator guidance: {instr}" if instr else ""


def _effective_topic(state: "StudyState") -> str:
    """Topic to show downstream agents: the orchestrator may have just inferred it
    from an uploaded PDF, so prefer the freshest value in orchestrator_plan if the
    original input topic was blank.
    """
    topic = (state.get("topic") or "").strip()
    if topic:
        return topic
    inferred = ((state.get("orchestrator_plan") or {}).get("topic") or "").strip()
    return inferred or "Uploaded document"


# ============================================================
# AGENT 1 — CONCEPT EXPLAINER
# ============================================================

def concept_agent(state: "StudyState") -> dict:
    system_prompt = (
        "You are a concept-explanation agent for exam preparation. Explain the given "
        "topic clearly and precisely, at the level expected in a university exam answer. "
        "Include one worked example. Use clear structure (short paragraphs, no filler). "
        "If source material is provided, ground your explanation in it and prefer its "
        "terminology and framing over generic textbook phrasing."
    )
    user_prompt = f"Topic: {_effective_topic(state)}{_plan_instruction(state, 'concept')}{_material_block(state)}"
    return {"concept_explanation": call_llm("concept", system_prompt, user_prompt)}


# ============================================================
# AGENT 2 — QUESTION GENERATOR
# ============================================================

def question_agent(state: "StudyState") -> dict:
    system_prompt = (
        "You are a practice-question generation agent for exam preparation. "
        'Respond ONLY with JSON: {"questions": [{"question": "...", "difficulty": '
        '"easy|medium|hard"}, ...]}. Produce exactly 5 practice questions covering '
        "the topic across all three difficulty tiers, mixing conceptual and "
        "problem-solving question types as appropriate for the subject. If source "
        "material is provided, base questions on it directly (facts, definitions, "
        "examples, or problems it actually contains) rather than generic questions."
    )
    user_prompt = (
        f"Topic: {_effective_topic(state)}{_plan_instruction(state, 'questions')}\n\n"
        f"Concept explanation for reference:\n{state.get('concept_explanation', '')}"
        f"{_material_block(state)}"
    )
    try:
        raw = call_llm("questions", system_prompt, user_prompt)
        result = _extract_json(raw, {"questions": []})
        questions = result.get("questions", [])
    except Exception as e:
        log.warning(f"Question generation failed: {e}")
        questions = []
    return {"questions": questions}


# ============================================================
# AGENT 3 — SOLUTIONS
# ============================================================

def solution_agent(state: "StudyState") -> dict:
    questions = state.get("questions", [])
    if not questions:
        return {"solutions": "No practice questions were generated, so no solutions to provide."}

    q_block = "\n".join(f"{i}. [{q.get('difficulty','?')}] {q.get('question','')}" for i, q in enumerate(questions, 1))
    system_prompt = (
        "You are a solutions agent for exam preparation. For each numbered practice "
        "question given, provide a full worked solution or model answer, and a short "
        "grading rubric (what a correct answer must include). Be exam-accurate — "
        "never guess at facts you're not confident in. If source material is provided, "
        "verify answers against it wherever the question draws on it."
    )
    user_prompt = (
        f"Topic: {_effective_topic(state)}{_plan_instruction(state, 'solutions')}\n\n"
        f"Questions:\n{q_block}{_material_block(state)}"
    )
    return {"solutions": call_llm("solutions", system_prompt, user_prompt, temperature=0.2)}


# ============================================================
# AGENT 4 — WEAK-AREA DIAGNOSTIC
# ============================================================

def weak_area_agent(state: "StudyState") -> dict:
    system_prompt = (
        "You are a weak-area diagnostic agent for exam preparation. Given a topic and "
        "a student's self-rated confidence (1-5), list the most common mistakes and "
        "misconceptions students have with this topic, and which specific sub-areas "
        "deserve extra attention given the stated confidence level. Be concrete and specific."
    )
    user_prompt = (
        f"Topic: {_effective_topic(state)}\n"
        f"Self-rated confidence: {state.get('confidence', 3)}/5"
        f"{_plan_instruction(state, 'weak_areas')}"
    )
    return {"weak_areas": call_llm("weak_areas", system_prompt, user_prompt)}


# ============================================================
# AGENT 5 — STUDY PLAN
# ============================================================

def study_plan_agent(state: "StudyState") -> dict:
    system_prompt = (
        "You are a study-planning agent. Given a topic, days remaining until the exam, "
        "and known weak areas, produce a day-by-day revision schedule. Be realistic about "
        "how much can be covered per day. If days remaining is unknown, produce a generic "
        "5-day plan instead."
    )
    user_prompt = (
        f"Topic: {_effective_topic(state)}\n"
        f"Days until exam: {state.get('days_until_exam', 'not specified')}\n"
        f"Weak areas identified:\n{state.get('weak_areas', '')}"
        f"{_plan_instruction(state, 'study_plan')}"
    )
    return {"study_plan": call_llm("study_plan", system_prompt, user_prompt)}


# ============================================================
# LANGGRAPH ORCHESTRATION
# ============================================================

class StudyState(TypedDict, total=False):
    topic: str
    confidence: int
    days_until_exam: int
    source_material: str
    source_material_name: str
    orchestrator_plan: dict
    concept_explanation: str
    questions: list[dict]
    solutions: str
    weak_areas: str
    study_plan: str
    run_started_at: str


def build_graph():
    """Fan-out / fan-in graph: after the orchestrator, the concept->questions->solutions
    branch and the weak_areas->study_plan branch have no data dependency on each other,
    so LangGraph schedules and runs them concurrently instead of strictly one-after-another.
    Combined with the shared rate limiter above, this is the main speed win - roughly
    the runtime of the longer of the two branches instead of the sum of all six.
    """
    graph = StateGraph(StudyState)
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("concept_agent", concept_agent)
    graph.add_node("question_agent", question_agent)
    graph.add_node("solution_agent", solution_agent)
    graph.add_node("weak_area_agent", weak_area_agent)
    graph.add_node("study_plan_agent", study_plan_agent)
    graph.set_entry_point("orchestrator")

    # fan-out: two independent branches start right after the orchestrator
    graph.add_edge("orchestrator", "concept_agent")
    graph.add_edge("orchestrator", "weak_area_agent")

    # branch A: concept -> questions -> solutions
    graph.add_edge("concept_agent", "question_agent")
    graph.add_edge("question_agent", "solution_agent")

    # branch B: weak_areas -> study_plan
    graph.add_edge("weak_area_agent", "study_plan_agent")

    # fan-in: both branches independently reach END
    graph.add_edge("solution_agent", END)
    graph.add_edge("study_plan_agent", END)
    return graph.compile()


# ============================================================
# STYLING — "Study Atelier" theme (indigo/violet, distinct from the lit-review app)
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #14121F;
    --surface: #1D1A2E;
    --surface-alt: #262239;
    --rule: #332E4A;
    --ink: #F1EEFB;
    --muted: #948FB0;
    --violet: #A78BFA;
    --violet-dim: #6D5FA3;
    --success: #6FBF8B;
    --danger: #D9755F;
    --easy: #6FBF8B;
    --medium: #E0B85C;
    --hard: #D9755F;
}

.stApp { background: var(--bg); color: var(--ink); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] * { color: var(--ink) !important; }

.masthead { border-bottom: 2px solid var(--violet); padding: 1.6rem 0 1.2rem 0; margin-bottom: 1.8rem; }
.masthead .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.18em;
    color: var(--violet); text-transform: uppercase;
}
.masthead h1 { font-size: 2.4rem; font-style: italic; font-weight: 600; margin: 0.2rem 0 0.3rem 0; }
.masthead .sub { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--muted); }

.stepper { display: flex; gap: 0; margin: 0.4rem 0 1.6rem 0; flex-wrap: wrap; }
.step {
    flex: 1; min-width: 100px; padding: 0.7rem; border-top: 3px solid var(--rule);
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--muted);
}
.step .num { color: var(--violet-dim); margin-right: 0.3rem; }
.step.active { border-top-color: var(--violet); color: var(--ink); }
.step.active .num { color: var(--violet); }
.step.done { border-top-color: var(--success); color: var(--ink); }
.step.done .num { color: var(--success); }

.panel {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.panel h3 { font-size: 1.15rem; margin-top: 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--rule); }

.q-card {
    background: var(--surface); border: 1px solid var(--rule); border-left: 3px solid var(--violet);
    border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
}
.q-card .q-num { font-family: 'JetBrains Mono', monospace; color: var(--violet); font-weight: 600; margin-right: 0.5rem; }
.diff-badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    padding: 0.15rem 0.5rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;
    margin-left: 0.5rem;
}
.diff-easy { background: rgba(111,191,139,0.15); color: var(--easy); border: 1px solid var(--easy); }
.diff-medium { background: rgba(224,184,92,0.15); color: var(--medium); border: 1px solid var(--medium); }
.diff-hard { background: rgba(217,117,95,0.15); color: var(--danger); border: 1px solid var(--danger); }

.stButton > button {
    background: var(--violet); color: #14121F; border: none; border-radius: 6px;
    font-weight: 600; padding: 0.55rem 1.4rem;
}
.stButton > button:hover { background: #BBA3FC; color: #14121F; }
[data-testid="stDownloadButton"] > button {
    background: var(--surface-alt); color: var(--ink); border: 1px solid var(--violet-dim); border-radius: 6px;
}

.footer-note {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--muted);
    text-align: center; padding: 1.5rem 0 0.5rem 0; border-top: 1px solid var(--rule); margin-top: 2rem;
}
</style>
"""

PIPELINE_STAGES = [
    ("00", "orchestrator", "Orchestrator"),
    ("01", "concept", "Concept"),
    ("02", "questions", "Questions"),
    ("03", "solutions", "Solutions"),
    ("04", "weak_areas", "Weak Areas"),
    ("05", "study_plan", "Study Plan"),
]


def render_stepper(active_stage: str):
    order = [s[1] for s in PIPELINE_STAGES]
    active_idx = order.index(active_stage) if active_stage in order else -1
    html = ['<div class="stepper">']
    for i, (num, key, label) in enumerate(PIPELINE_STAGES):
        cls = "step"
        if i < active_idx:
            cls += " done"
        elif i == active_idx:
            cls += " active"
        html.append(f'<div class="{cls}"><span class="num">{num}</span>{label}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ============================================================
# STREAMLIT APP
# ============================================================

def build_markdown_report(final_state: dict) -> str:
    questions = final_state.get("questions", [])
    q_block = "\n".join(
        f"{i}. [{q.get('difficulty','?')}] {q.get('question','')}" for i, q in enumerate(questions, 1)
    )
    source_note = ""
    if final_state.get("source_material_name"):
        source_note = f"_Source document: {final_state['source_material_name']}_\n\n"
    return (
        f"# Study Pack: {final_state.get('topic', '')}\n\n"
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
        f"{source_note}"
        f"## Concept Explanation\n\n{final_state.get('concept_explanation', '')}\n\n"
        f"## Practice Questions\n\n{q_block}\n\n"
        f"## Solutions\n\n{final_state.get('solutions', '')}\n\n"
        f"## Weak Areas to Watch\n\n{final_state.get('weak_areas', '')}\n\n"
        f"## Study Plan\n\n{final_state.get('study_plan', '')}\n"
    )


def main():
    st.set_page_config(page_title="StudyForge AI", page_icon="🎓", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.session_state.setdefault("final_state", None)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("history", [])

    # Defaults for the form widgets below live here, set ONCE via setdefault. The
    # widgets themselves are then created with no separate `value=`/`index=` default,
    # since Streamlit raises a "created with a default value but also had its value
    # set via the Session State API" error/warning when a widget both declares its
    # own default AND has its session_state key written elsewhere (e.g. the Clear
    # button's callback below). Only ever touching the key through session_state
    # (here and in the callback) avoids that conflict entirely.
    st.session_state.setdefault("topic_input", "")
    st.session_state.setdefault("confidence_input", 3)
    st.session_state.setdefault("exam_date_input", None)

    # ---------- sidebar ----------
    with st.sidebar:
        st.markdown("### Configuration")

        key_configured = bool(resolve_api_key())

        if key_configured:
            st.success("API key loaded.")
        else:
            st.error(
                "No OpenRouter API key configured.\n\n"
                "Set OPENROUTER_API_KEY in .streamlit/secrets.toml or a .env file, "
                "then restart the app.\n"
                "Get a key at openrouter.ai/keys"
            )

        st.markdown("---")
        st.caption("Orchestrator + 5 agents, each a free OpenRouter model with automatic fallback.")
        with st.expander("Agents & models"):
            for agent, chain in AGENT_MODEL_FALLBACKS.items():
                st.markdown(f"**{agent}**")
                st.caption(" → ".join(chain))

        st.markdown("---")
        st.markdown("### History")
        history = st.session_state["history"]
        if not history:
            st.caption("No past runs yet.")
        else:
            for i, entry in enumerate(history):
                with st.expander(f"{entry['topic']} — {entry['timestamp']}"):
                    st.caption(f"Confidence: {entry['confidence']}/5")
                    if entry.get("days_until_exam") is not None:
                        st.caption(f"Days until exam: {entry['days_until_exam']}")
                    if st.button("View this run", key=f"view_history_{i}"):
                        st.session_state["final_state"] = entry["state"]
                        st.rerun()
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()

    # ---------- masthead ----------
    st.markdown(
        """
        <div class="masthead">
            <div class="eyebrow">Agentic Study Tooling · No. 01</div>
            <h1>StudyForge AI</h1>
            <div class="sub">Orchestrator → Concept → Questions → Solutions → Weak Areas → Study Plan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        topic = st.text_input(
            "Study topic (optional if you upload a PDF below)",
            placeholder="e.g. NFA to DFA conversion, pumping lemma, LR parsing",
            key="topic_input",
        )
    with col_b:
        confidence = st.select_slider("Confidence (1-5)", options=[1, 2, 3, 4, 5], key="confidence_input")
    with col_c:
        exam_date = st.date_input("Exam date (optional)", key="exam_date_input")

    uploaded_pdf = st.file_uploader(
        "Or upload lecture notes / a textbook chapter (PDF) — the pipeline will read it "
        "and ground the concept, questions, and solutions in its content",
        type=["pdf"],
        key="pdf_upload",
    )

    days_until_exam = None
    if exam_date:
        days_until_exam = (exam_date - date.today()).days

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            "Run pipeline", type="primary",
            disabled=st.session_state["running"] or not key_configured,
            use_container_width=True,
        )
    with col_clear:
        def _clear_form():
            st.session_state["topic_input"] = ""
            st.session_state["confidence_input"] = 3
            st.session_state["exam_date_input"] = None
            st.session_state["final_state"] = None

        st.button("Clear", on_click=_clear_form, use_container_width=True)

    if not key_configured:
        st.caption("⚠️ Add an OpenRouter API key via secrets.toml or .env to enable this.")

    stepper_placeholder = st.empty()

    if run_clicked:
        if not topic.strip() and uploaded_pdf is None:
            st.error("Enter a study topic or upload a PDF first.")
        else:
            source_material = None
            if uploaded_pdf is not None:
                try:
                    with st.spinner("Reading PDF..."):
                        source_material = extract_pdf_text(uploaded_pdf)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

            st.session_state["running"] = True
            app = build_graph()
            state = {"topic": topic, "confidence": confidence}
            if days_until_exam is not None:
                state["days_until_exam"] = days_until_exam
            if source_material:
                state["source_material"] = source_material
                state["source_material_name"] = uploaded_pdf.name
            # Seed with the initial input state (topic/confidence/days_until_exam/
            # source_material) and accumulate each node's partial update into it.
            # Nodes now only return the keys they change (required so concurrent
            # branches don't collide on shared keys), so this dict has to be built
            # up across the stream rather than replaced wholesale on every event.
            final_state = dict(state)
            active_stage = "orchestrator"
            completed_stages = set()

            try:
                with stepper_placeholder.container():
                    render_stepper(active_stage)
                with st.status("Running agent pipeline...", expanded=True) as status:
                    for event in app.stream(state):
                        # app.stream() can yield more than one node per superstep now
                        # that concept/weak_areas (and questions/study_plan, etc.) run
                        # concurrently, so handle every node in the event, not just one.
                        for node_name, node_state in event.items():
                            if node_name == "orchestrator":
                                completed_stages.add("orchestrator")
                                n_focus = len((node_state.get("orchestrator_plan") or {}).get("focus_areas", []))
                                inferred_topic = node_state.get("topic")
                                msg = f"🧭 Orchestrator Agent — plan drafted, {n_focus} focus areas identified"
                                if inferred_topic:
                                    msg += f" (topic inferred from PDF: {inferred_topic})"
                                st.write(msg)
                            elif node_name == "concept_agent":
                                completed_stages.add("concept")
                                st.write("📘 Concept Explainer Agent — explanation written")
                            elif node_name == "question_agent":
                                completed_stages.add("questions")
                                st.write(f"❓ Question Generator Agent — {len(node_state.get('questions', []))} questions written")
                            elif node_name == "solution_agent":
                                completed_stages.add("solutions")
                                st.write("✅ Solution Agent — worked solutions written")
                            elif node_name == "weak_area_agent":
                                completed_stages.add("weak_areas")
                                st.write("🎯 Weak-Area Diagnostic Agent — pitfalls identified")
                            elif node_name == "study_plan_agent":
                                completed_stages.add("study_plan")
                                st.write("🗓️ Study Plan Agent — revision schedule built")
                            final_state.update(node_state)
                            order = [s[1] for s in PIPELINE_STAGES]
                            active_stage = next((s for s in order if s not in completed_stages), order[-1])
                            with stepper_placeholder.container():
                                render_stepper(active_stage)
                    status.update(label="Pipeline complete", state="complete")

                st.session_state["final_state"] = final_state
                if final_state:
                    st.session_state["history"].insert(0, {
                        "topic": final_state.get("topic", topic),
                        "confidence": confidence,
                        "days_until_exam": days_until_exam,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "state": final_state,
                    })
                    st.session_state["history"] = st.session_state["history"][:20]
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                log.exception("Pipeline error")
            finally:
                st.session_state["running"] = False

    # ---------- results ----------
    final_state = st.session_state.get("final_state")
    if final_state:
        if final_state.get("source_material_name"):
            st.caption(f"📄 Grounded in uploaded document: **{final_state['source_material_name']}**")

        tab_concept, tab_q, tab_sol, tab_weak, tab_plan = st.tabs(
            ["Concept", f"Questions ({len(final_state.get('questions', []))})", "Solutions", "Weak Areas", "Study Plan"]
        )

        with tab_concept:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(final_state.get("concept_explanation", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_q:
            for i, q in enumerate(final_state.get("questions", []), 1):
                diff = q.get("difficulty", "medium")
                st.markdown(
                    f"""
                    <div class="q-card">
                        <span class="q-num">Q{i}.</span>{q.get('question', '')}
                        <span class="diff-badge diff-{diff}">{diff}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_sol:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(final_state.get("solutions", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_weak:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(final_state.get("weak_areas", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_plan:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(final_state.get("study_plan", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.download_button(
            "Download full study pack (.md)",
            data=build_markdown_report(final_state),
            file_name=f"study_pack_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    st.markdown(
        '<div class="footer-note">StudyForge AI · orchestrator + 5 free-tier agents</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()