"""FastAPI chatbot API - session-based Q&A with class/subject set once per chat."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException

load_dotenv()
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_config
from ..rag.tutor import Tutor

# In-memory session store: session_id -> {class_, subject, chapter}
_sessions: dict[str, dict] = {}

# Default contexts if Chroma is empty
_DEFAULT_CONTEXTS = [
    {"class": "10", "subject": "Science"},
    {"class": "10", "subject": "Mathematics"},
    {"class": "9", "subject": "Science"},
    {"class": "9", "subject": "Mathematics"},
]


def _get_available_contexts() -> list[dict]:
    """Get available class/subject combinations from Chroma metadata."""
    try:
        from ..vectorstore import ChromaStore

        store = ChromaStore(get_config())
        col = store.collection()
        result = col.get(include=["metadatas"], limit=5000)
        metas = result.get("metadatas") or []
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for m in metas:
            if m:
                c = str(m.get("class", "")).strip()
                s = str(m.get("subject", "")).strip()
                if c and s and (c, s) not in seen:
                    seen.add((c, s))
                    out.append({"class": c, "subject": s})
        if out:
            return sorted(out, key=lambda x: (x["class"], x["subject"]))
    except Exception:
        pass
    return _DEFAULT_CONTEXTS


app = FastAPI(title="AI Tutor Chatbot", version="1.0.0")


# --- Pydantic models ---


class StartChatRequest(BaseModel):
    class_: str = Field(..., alias="class", description="Class (e.g., 10)")
    subject: str = Field(..., description="Subject (e.g., Science)")
    chapter: str | None = Field(None, description="Optional chapter filter")

    class Config:
        populate_by_name = True


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Student question")


class ChatResponse(BaseModel):
    session_id: str
    message: str


class AnswerResponse(BaseModel):
    answer: str


# --- API routes ---


@app.get("/api/contexts")
def get_contexts() -> list[dict]:
    """List available class/subject combinations from indexed NCERT content."""
    return _get_available_contexts()


@app.post("/api/chat/start", response_model=ChatResponse)
def start_chat(req: StartChatRequest) -> ChatResponse:
    """Start a new chat session. Class and subject are set once for the entire chat."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "class_": str(req.class_).strip(),
        "subject": str(req.subject).strip(),
        "chapter": str(req.chapter).strip() if req.chapter else None,
    }
    return ChatResponse(
        session_id=session_id,
        message=f"Chat started for Class {req.class_}, {req.subject}. Ask your questions!",
    )


@app.post("/api/chat/{session_id}/ask", response_model=AnswerResponse)
def ask_question(session_id: str, req: AskRequest) -> AnswerResponse:
    """Ask a question in an existing chat. Uses the session's class/subject."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start a new chat.")
    ctx = _sessions[session_id]
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set. Create a .env file in the project root with GEMINI_API_KEY=your_key and restart the server.",
        )
    if not os.environ.get("GEMINI_MODEL"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_MODEL not set. Add GEMINI_MODEL=your_model to .env (e.g. models/gemini-2.5-flash) and restart.",
        )
    try:
        tutor = Tutor()
        answer = tutor.answer(
            query=req.query.strip(),
            class_=ctx["class_"],
            subject=ctx["subject"],
            chapter=ctx["chapter"],
            top_k=5,
        )
        return AnswerResponse(answer=answer)
    except (KeyError, ValueError) as e:
        msg = str(e)
        if "GEMINI" in msg or "GEMINI_API_KEY" in msg or "GEMINI_MODEL" in msg:
            raise HTTPException(
                status_code=503,
                detail=msg or "GEMINI_API_KEY and GEMINI_MODEL must be set in .env",
            ) from e
        raise


@app.get("/api/chat/{session_id}/context")
def get_session_context(session_id: str) -> dict:
    """Get the class/subject/chapter for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    ctx = _sessions[session_id].copy()
    return {"class": ctx["class_"], "subject": ctx["subject"], "chapter": ctx["chapter"]}


# --- Serve chatbot UI ---

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>AI Tutor Chatbot</h1><p>Place index.html in api/static/</p>")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
