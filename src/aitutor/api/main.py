"""FastAPI chatbot API - session-based Q&A with class/subject set once per chat."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from google.genai.errors import ClientError, ServerError

# Load `.env` from repo root (uvicorn cwd may not be the project folder).
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env", override=True)

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_config, get_gemini_api_key


def _raise_if_gemini_api_key_invalid(exc: BaseException) -> None:
    """Map Google's API_KEY_INVALID to a clear HTTP error (avoids a 500 + traceback)."""
    if isinstance(exc, ClientError):
        msg = str(exc)
        if "API_KEY_INVALID" in msg or "api key not valid" in msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini rejected your API key (API_KEY_INVALID). "
                    "Create a Developer API key at https://aistudio.google.com/apikey . "
                    "If the key comes from Google Cloud Console: use Application restrictions = "
                    "'None' or 'IP addresses' (not 'HTTP referrers' — that breaks server apps), and "
                    "allow the Generative Language API. Put the key in `.env` as GEMINI_API_KEY=... "
                    "and restart uvicorn."
                ),
            ) from exc
from ..rag.tutor import Tutor
from ..generation import ClarifierLLM, SummaryLLM

# In-memory session store:
# session_id -> {class_, subject, chapter, last_question, last_answer, history}
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


def _is_clarification_request(text: str) -> bool:
    """Heuristic: detect when the student is asking to explain again / in a better way."""
    t = text.strip().lower()
    if not t:
        return False
    keywords = [
        "could not understand",
        "can't understand",
        "cannot understand",
        "didn't understand",
        "did not understand",
        "explain again",
        "explain in a better way",
        "more clearly",
        "explain once again",
        "answer in an easier war",
        "answer in a simpler way",
        "answer in a more detailed way",
        "answer in a more concise way",
        "answer in a more clear way",
        "answer in a more understandable way",
        "answer in a more easy to understand way",
        "answer in a more easy to understand way",
        "explain it better",
        "explain in detail",
        "explain this in detail",
        "explain in more detail",
        "tell in detail",
        "simpler way",
        "simple way",
        "step by step",
        "more examples",
        "give me some more examples",
        "give some examples",
        "some examples",
        "explain this with example",
        "explain this with examples",
        "explain this by giving example",
        "explain this by giving examples",
        "explain with example",
        "explain with examples",
        "another example",
        "one more example",
        "solve again",
        "solve it again",
    ]
    return any(k in t for k in keywords)


def _is_summary_request(text: str) -> bool:
    """Heuristic: detect when the student is asking for a summary/short notes."""
    t = text.strip().lower()
    if not t:
        return False
    phrases = [
        "summarise",
        "summarize",
        "give a summary",
        "give me a summary",
        "short notes",
        "in short",
        "in brief",
        "briefly explain",
        "explain in short",
        "explain briefly",
        "make short notes",
        "summary of this",
        "summary of the topic",
    ]
    return any(p in t for p in phrases)


def _append_history(ctx: dict, *, role: str, text: str) -> None:
    if "history" not in ctx or not isinstance(ctx.get("history"), list):
        ctx["history"] = []
    ctx["history"].append({"role": role, "text": text})


def _is_rag_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return False
    triggers = [
        "i can’t answer this confidently from the retrieved ncert content",
        "i can't answer this confidently from the retrieved ncert content",
        "i can’t answer this from the ncert text i have indexed",
        "i can't answer this from the ncert text i have indexed",
        "low retrieval confidence",
    ]
    return any(t in a for t in triggers)


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
        "last_question": None,
        "last_answer": None,
        "history": [],
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
    user_query = req.query.strip()
    if not get_gemini_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Add it to `.env` in the project root "
                "and restart the server."
            ),
        )
    # GEMINI_MODEL defaults to models/gemini-2.5-flash; set GEMINI_MODEL in .env to override.

    # If the student asks to summarise a topic and we have a previous
    # explanation or some text in the thread, use a cloud LLM to produce
    # a level-appropriate summary instead of going through RAG again.
    if _is_summary_request(user_query) and ctx.get("last_answer"):
        try:
            summariser = SummaryLLM.default()
            _append_history(ctx, role="user", text=user_query)
            source_text = ctx["last_answer"]
            topic_hint = ctx.get("last_question") or user_query
            answer = summariser.summarize(
                class_=ctx["class_"],
                subject=ctx["subject"],
                chapter=ctx["chapter"],
                topic_hint=topic_hint,
                source_text=source_text,
                summary_request=user_query,
            )
        except (KeyError, ValueError, ClientError, ServerError) as e:
            _raise_if_gemini_api_key_invalid(e)
            msg = str(e)
            if "GEMINI" in msg or "GEMINI_API_KEY" in msg or "GEMINI_MODEL" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=msg or "GEMINI_API_KEY and GEMINI_MODEL must be set in .env",
                ) from e
            if isinstance(e, ClientError) and "RESOURCE_EXHAUSTED" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The tutor has temporarily hit its usage limit for the Gemini model. "
                        "Please wait a minute and try your question again."
                    ),
                ) from e
            if isinstance(e, ServerError) and "UNAVAILABLE" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The Gemini model is temporarily overloaded (high demand). "
                        "Please try again in a minute."
                    ),
                ) from e
            raise
        else:
            ctx["last_answer"] = answer
            _append_history(ctx, role="assistant", text=answer)
            _sessions[session_id] = ctx
            return AnswerResponse(answer=answer)

    # If the student is not satisfied and asks to explain/solve again,
    # use a cloud LLM fallback that re-explains the previous answer.
    if _is_clarification_request(user_query) and ctx.get("last_question") and ctx.get("last_answer"):
        try:
            clarifier = ClarifierLLM.default()
            _append_history(ctx, role="user", text=user_query)
            answer = clarifier.clarify(
                class_=ctx["class_"],
                subject=ctx["subject"],
                chapter=ctx["chapter"],
                question=ctx["last_question"],
                previous_answer=ctx["last_answer"],
                student_followup=user_query,
                thread=ctx.get("history") or [],
            )
        except (KeyError, ValueError, ClientError, ServerError) as e:
            _raise_if_gemini_api_key_invalid(e)
            msg = str(e)
            if "GEMINI" in msg or "GEMINI_API_KEY" in msg or "GEMINI_MODEL" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=msg or "GEMINI_API_KEY and GEMINI_MODEL must be set in .env",
                ) from e
            if isinstance(e, ClientError) and "RESOURCE_EXHAUSTED" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The tutor has temporarily hit its usage limit for the Gemini model. "
                        "Please wait a minute and then ask again, or try a shorter question."
                    ),
                ) from e
            if isinstance(e, ServerError) and "UNAVAILABLE" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The Gemini model is temporarily overloaded (high demand). "
                        "Please try again in a minute."
                    ),
                ) from e
            raise
        else:
            ctx["last_answer"] = answer
            _append_history(ctx, role="assistant", text=answer)
            _sessions[session_id] = ctx
            return AnswerResponse(answer=answer)

    try:
        tutor = Tutor()
        _append_history(ctx, role="user", text=user_query)
        answer = tutor.answer(
            query=user_query,
            class_=ctx["class_"],
            subject=ctx["subject"],
            chapter=ctx["chapter"],
            top_k=5,
        )

        # If RAG refuses due to missing/low-confidence NCERT context and the student
        # is explicitly asking for re-explanation/more examples, fall back to cloud LLM
        # and keep output at the student's class level.
        if _is_rag_refusal(answer) and _is_clarification_request(user_query):
            try:
                clarifier = ClarifierLLM.default()
                answer = clarifier.clarify(
                    class_=ctx["class_"],
                    subject=ctx["subject"],
                    chapter=ctx["chapter"],
                    question=ctx.get("last_question") or user_query,
                    previous_answer=ctx.get("last_answer") or answer,
                    student_followup=user_query,
                    thread=ctx.get("history") or [],
                )
            except (KeyError, ValueError, ClientError, ServerError) as e:
                _raise_if_gemini_api_key_invalid(e)
                msg = str(e)
                if "GEMINI" in msg or "GEMINI_API_KEY" in msg or "GEMINI_MODEL" in msg:
                    raise HTTPException(
                        status_code=503,
                        detail=msg or "GEMINI_API_KEY and GEMINI_MODEL must be set in .env",
                    ) from e
                if isinstance(e, ClientError) and "RESOURCE_EXHAUSTED" in msg:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "The tutor has temporarily hit its usage limit for the Gemini model. "
                            "Please wait a minute and try again, or rephrase your question."
                        ),
                    ) from e
                if isinstance(e, ServerError) and "UNAVAILABLE" in msg:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "The Gemini model is temporarily overloaded (high demand). "
                            "Please try again in a minute."
                        ),
                    ) from e
                raise

        ctx["last_question"] = user_query
        ctx["last_answer"] = answer
        _append_history(ctx, role="assistant", text=answer)
        _sessions[session_id] = ctx
        return AnswerResponse(answer=answer)
    except (KeyError, ValueError, ClientError, ServerError) as e:
        _raise_if_gemini_api_key_invalid(e)
        msg = str(e)
        if "GEMINI" in msg or "GEMINI_API_KEY" in msg or "GEMINI_MODEL" in msg:
            raise HTTPException(
                status_code=503,
                detail=msg or "GEMINI_API_KEY and GEMINI_MODEL must be set in .env",
            ) from e
        if isinstance(e, ClientError) and "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The tutor has temporarily hit its usage limit for the Gemini model. "
                    "Please wait a minute and try again."
                ),
            ) from e
        if isinstance(e, ServerError) and "UNAVAILABLE" in msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Gemini model is temporarily overloaded (high demand). "
                    "Please try again in a minute."
                ),
            ) from e
        raise


@app.get("/api/chat/{session_id}/context")
def get_session_context(session_id: str) -> dict:
    """Get the class/subject/chapter for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    ctx = _sessions[session_id].copy()
    return {"class": ctx["class_"], "subject": ctx["subject"], "chapter": ctx["chapter"]}


@app.get("/api/chat/{session_id}/history")
def get_session_history(session_id: str) -> dict:
    """Get the full message thread for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    ctx = _sessions[session_id]
    return {"session_id": session_id, "history": ctx.get("history") or []}


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
