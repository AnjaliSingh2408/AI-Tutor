"""FastAPI chatbot API - session-based Q&A with class/subject set once per chat."""

from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field

from ..config import get_config, get_gemini_api_key
from ..rag.tutor import Tutor
from ..generation import ClarifierLLM, SummaryLLM

# --- Load .env from repo root ---
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env", override=True)

# --- App ---
app = FastAPI(title="AI Tutor Chatbot", version="1.0.0")

# --- Session store ---
_sessions: dict[str, dict] = {}

_DEFAULT_CONTEXTS = [
    {"class": "10", "subject": "Science"},
    {"class": "10", "subject": "Mathematics"},
    {"class": "9", "subject": "Science"},
    {"class": "9", "subject": "Mathematics"},
]


# ---------- Utilities ----------

def _raise_if_gemini_api_key_invalid(exc: BaseException) -> None:
    if isinstance(exc, ClientError):
        msg = str(exc)
        if "API_KEY_INVALID" in msg or "api key not valid" in msg.lower():
            raise HTTPException(
                status_code=503,
                detail="Invalid Gemini API key. Fix GEMINI_API_KEY in .env",
            ) from exc


def _get_available_contexts() -> list[dict]:
    try:
        from ..vectorstore import ChromaStore

        store = ChromaStore(get_config())
        col = store.collection()
        result = col.get(include=["metadatas"], limit=5000)

        metas = result.get("metadatas") or []
        seen = set()
        out = []

        for m in metas:
            if m:
                c = str(m.get("class", "")).strip()
                s = str(m.get("subject", "")).strip()
                if c and s and (c, s) not in seen:
                    seen.add((c, s))
                    out.append({"class": c, "subject": s})

        return sorted(out, key=lambda x: (x["class"], x["subject"])) or _DEFAULT_CONTEXTS

    except Exception:
        return _DEFAULT_CONTEXTS


def _append_history(ctx: dict, *, role: str, text: str) -> None:
    ctx.setdefault("history", []).append({"role": role, "text": text})


def _is_summary_request(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["summary", "summarise", "short notes", "in brief"])


def _is_clarification_request(text: str) -> bool:
    t = text.lower()
    return any(
        p in t
        for p in [
            "explain again",
            "didn't understand",
            "simpler",
            "more examples",
            "step by step",
        ]
    )


# ---------- Pydantic models ----------

class StartChatRequest(BaseModel):
    class_: str = Field(..., alias="class")
    subject: str
    chapter: str | None = None

    class Config:
        populate_by_name = True


class AskRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    session_id: str
    message: str


class AnswerResponse(BaseModel):
    answer: str


# ---------- Routes ----------

@app.get("/api/contexts")
def get_contexts():
    return _get_available_contexts()


@app.post("/api/chat/start", response_model=ChatResponse)
def start_chat(req: StartChatRequest):
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "class_": req.class_,
        "subject": req.subject,
        "chapter": req.chapter,
        "last_question": None,
        "last_answer": None,
        "history": [],
    }

    return ChatResponse(
        session_id=session_id,
        message=f"Chat started for Class {req.class_}, {req.subject}",
    )


@app.post("/api/chat/{session_id}/ask", response_model=AnswerResponse)
def ask_question(session_id: str, req: AskRequest):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")

    if not get_gemini_api_key():
        raise HTTPException(503, "GEMINI_API_KEY missing in .env")

    ctx = _sessions[session_id]
    user_query = req.query.strip()

    # --- SUMMARY MODE ---
    if _is_summary_request(user_query) and ctx.get("last_answer"):
        summariser = SummaryLLM.default()
        answer = summariser.summarize(
            class_=ctx["class_"],
            subject=ctx["subject"],
            chapter=ctx["chapter"],
            topic_hint=ctx["last_question"] or user_query,
            source_text=ctx["last_answer"],
            summary_request=user_query,
        )

    # --- CLARIFICATION MODE ---
    elif _is_clarification_request(user_query) and ctx.get("last_answer"):
        clarifier = ClarifierLLM.default()
        answer = clarifier.clarify(
            class_=ctx["class_"],
            subject=ctx["subject"],
            chapter=ctx["chapter"],
            question=ctx["last_question"],
            previous_answer=ctx["last_answer"],
            student_followup=user_query,
            thread=ctx["history"],
        )

    # --- NORMAL RAG MODE ---
    else:
        tutor = Tutor.default()   # ⭐⭐⭐ FIXED LINE ⭐⭐⭐

        answer = tutor.answer(
            query=user_query,
            class_=ctx["class_"],
            subject=ctx["subject"],
            chapter=ctx["chapter"],
            top_k=5,
        )

    ctx["last_question"] = user_query
    ctx["last_answer"] = answer
    _append_history(ctx, role="assistant", text=answer)

    return AnswerResponse(answer=answer)


@app.get("/api/chat/{session_id}/context")
def get_context(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    ctx = _sessions[session_id]
    return {"class": ctx["class_"], "subject": ctx["subject"], "chapter": ctx["chapter"]}


@app.get("/api/chat/{session_id}/history")
def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    return {"history": _sessions[session_id]["history"]}


# ---------- UI ----------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>AI Tutor Chatbot</h1>")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")