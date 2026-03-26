from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mimetypes
import re
import time
import uuid

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from ..config import AppConfig, get_config, get_gemini_api_key
from ..ingest import extract_pdf_diagrams
from ..rag.tutor import Tutor
from ..vectorstore import ChromaStore


def _cosine_from_distance(distance: float) -> float:
    # Chroma with cosine space: distance ~= 1 - cosine_similarity
    return 1.0 - float(distance)


def _is_quota_exhausted_error(exc: ClientError) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "quota exceeded" in msg.lower()


def _is_daily_quota_error(exc: ClientError) -> bool:
    return "PerDay" in str(exc) or "per day" in str(exc).lower()


def _extract_first_embedding_vector(resp: object) -> list[float]:
    embeddings = getattr(resp, "embeddings", None) or []
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return [float(x) for x in values]
    embedding = getattr(resp, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return [float(x) for x in values]
    raise ValueError("No embedding vector returned by Gemini embedding API.")


@dataclass
class MultiModalTutor:
    cfg: AppConfig
    store: ChromaStore
    qa_tutor: Tutor

    @classmethod
    def default(cls) -> "MultiModalTutor":
        cfg = get_config()
        store = ChromaStore(cfg)
        return cls(cfg=cfg, store=store, qa_tutor=Tutor.default())

    def _client(self) -> genai.Client:
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")
        return genai.Client(api_key=api_key)

    def _sleep_for_quota_retry(self, exc: ClientError, *, fallback_seconds: float = 45.0) -> None:
        msg = str(exc)
        # API error typically includes "... Please retry in 40.910800794s."
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
        wait_for = float(match.group(1)) if match else fallback_seconds
        wait_for = max(2.0, wait_for + 1.0)
        print(f"[MM-INDEX] quota hit (429), waiting {wait_for:.1f}s before retry...")
        time.sleep(wait_for)

    def _embed_text(self, text: str) -> list[float]:
        client = self._client()
        for attempt in range(1, 7):
            try:
                resp = client.models.embed_content(
                    model=self.cfg.gemini_embed_model,
                    contents=[text],
                )
                return _extract_first_embedding_vector(resp)
            except ClientError as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) and attempt < 6:
                    self._sleep_for_quota_retry(exc)
                    continue
                raise

    def _embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client()
        for attempt in range(1, 7):
            try:
                resp = client.models.embed_content(
                    model=self.cfg.gemini_embed_model,
                    contents=texts,
                )
                embeddings = getattr(resp, "embeddings", None) or []
                out: list[list[float]] = []
                for e in embeddings:
                    values = getattr(e, "values", None)
                    if values is None:
                        continue
                    out.append([float(x) for x in values])
                if len(out) != len(texts):
                    # Fallback for partial responses from provider.
                    return [self._embed_text(t) for t in texts]
                return out
            except ClientError as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) and attempt < 6:
                    self._sleep_for_quota_retry(exc)
                    continue
                raise
        return [self._embed_text(t) for t in texts]

    def _embed_bytes(self, *, blob: bytes, mime_type: str) -> list[float]:
        client = self._client()
        part = types.Part.from_bytes(data=blob, mime_type=mime_type)
        for attempt in range(1, 7):
            try:
                resp = client.models.embed_content(
                    model=self.cfg.gemini_embed_model,
                    contents=[part],
                )
                return _extract_first_embedding_vector(resp)
            except ClientError as exc:
                if "RESOURCE_EXHAUSTED" in str(exc) and attempt < 6:
                    self._sleep_for_quota_retry(exc)
                    continue
                raise

    def rebuild_multimodal_index(
        self,
        *,
        class_: str,
        subject: str,
        chapter: str | None = None,
        diagrams_dir: str | Path | None = None,
        extract_from_pdfs: bool = False,
        diagrams_cache_dir: str | Path | None = None,
        max_text_chunks: int = 5000,
        text_batch_size: int = 32,
        resume: bool = False,
    ) -> dict:
        if not resume:
            self.store.reset_multimodal()

        existing_text_chunk_ids: set[str] = set()
        existing_image_names: set[str] = set()
        if resume:
            filters = [
                {"class": {"$eq": str(class_)}},
                {"subject": {"$eq": str(subject)}},
            ]
            if chapter:
                filters.append({"chapter": {"$eq": str(chapter)}})
            where_existing = {"$and": filters}
            existing = self.store.multimodal_collection().get(
                where=where_existing,
                include=["metadatas"],
                limit=200000,
            )
            for m in (existing.get("metadatas") or []):
                meta = dict(m or {})
                cid = str(meta.get("source_chunk_id") or "").strip()
                if cid:
                    existing_text_chunk_ids.add(cid)
                sname = str(meta.get("source_image") or "").strip()
                if sname:
                    existing_image_names.add(sname)
            print(
                "[MM-INDEX] resume mode: "
                f"found {len(existing_text_chunk_ids)} text items and "
                f"{len(existing_image_names)} image items already indexed"
            )

        # 1) Index text chunks already stored in NCERT text collection.
        where_filters = [
            {"class": {"$eq": str(class_)}},
            {"subject": {"$eq": str(subject)}},
        ]
        if chapter:
            where_filters.append({"chapter": {"$eq": str(chapter)}})
        where = {"$and": where_filters}

        raw = self.store.get(
            where=where,
            limit=max_text_chunks,
            include=["documents", "metadatas"],
        )
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        mm_ids: list[str] = []
        mm_embeddings: list[list[float]] = []
        mm_docs: list[str] = []
        mm_metas: list[dict] = []

        text_rows: list[tuple[str, str, dict]] = []
        for cid, doc, meta in zip(ids, docs, metas):
            text = str(doc or "").strip()
            if text:
                text_rows.append((str(cid), text, dict(meta or {})))

        total = len(text_rows)
        filtered_rows = [row for row in text_rows if row[0] not in existing_text_chunk_ids]
        total_new = len(filtered_rows)
        for i in range(0, total_new, max(1, text_batch_size)):
            batch = filtered_rows[i : i + max(1, text_batch_size)]
            batch_texts = [row[1] for row in batch]
            try:
                batch_embeddings = self._embed_text_batch(batch_texts)
            except ClientError as exc:
                if _is_quota_exhausted_error(exc):
                    if _is_daily_quota_error(exc):
                        print("[MM-INDEX] Daily embedding quota exhausted. Stopping with partial index.")
                    else:
                        print("[MM-INDEX] Embedding quota exhausted. Stopping with partial index.")
                    break
                raise
            for (cid, text, meta), emb in zip(batch, batch_embeddings):
                mm_ids.append(f"text|{cid}")
                mm_embeddings.append(emb)
                mm_docs.append(text[:4000])
                meta["modality"] = "text"
                meta["source_chunk_id"] = str(cid)
                mm_metas.append(meta)
            print(f"[MM-INDEX] embedded text chunks: {min(i + len(batch), total_new)}/{total_new}")

        # 2) Optionally index diagram images (for diagram->text matching).
        image_count = 0
        extracted_from_pdf = 0
        cache_dir = Path(diagrams_cache_dir) if diagrams_cache_dir else (self.cfg.data_dir / "diagram_cache")

        if extract_from_pdfs and diagrams_dir:
            print(f"[MM-INDEX] extracting diagrams from PDFs under: {diagrams_dir}")
            extracted = extract_pdf_diagrams(
                books_dir=diagrams_dir,
                out_dir=cache_dir,
            )
            extracted_from_pdf = len(extracted)
            print(f"[MM-INDEX] extracted PDF diagrams: {extracted_from_pdf}")

        if diagrams_dir:
            ddir = Path(diagrams_dir)
            if ddir.exists():
                image_files = {
                    *ddir.rglob("*.png"),
                    *ddir.rglob("*.jpg"),
                    *ddir.rglob("*.jpeg"),
                    *ddir.rglob("*.webp"),
                }
                if cache_dir.exists():
                    image_files.update(cache_dir.rglob("*.png"))
                    image_files.update(cache_dir.rglob("*.jpg"))
                    image_files.update(cache_dir.rglob("*.jpeg"))
                    image_files.update(cache_dir.rglob("*.webp"))
                for img in image_files:
                    if img.name in existing_image_names:
                        continue
                    blob = img.read_bytes()
                    mime = mimetypes.guess_type(str(img))[0] or "image/png"
                    try:
                        emb = self._embed_bytes(blob=blob, mime_type=mime)
                    except ClientError as exc:
                        if _is_quota_exhausted_error(exc):
                            if _is_daily_quota_error(exc):
                                print("[MM-INDEX] Daily embedding quota exhausted while indexing images.")
                            else:
                                print("[MM-INDEX] Embedding quota exhausted while indexing images.")
                            break
                        raise
                    mm_ids.append(f"image|{uuid.uuid4().hex[:12]}")
                    mm_embeddings.append(emb)
                    mm_docs.append(f"Diagram file: {img.name}")
                    mm_metas.append(
                        {
                            "class": str(class_),
                            "subject": str(subject),
                            "chapter": str(chapter) if chapter else "",
                            "modality": "image",
                            "source_image": img.name,
                            "extracted_from_pdf": str(img).startswith(str(cache_dir)),
                        }
                    )
                    image_count += 1

        self.store.add_multimodal(
            ids=mm_ids,
            embeddings=mm_embeddings,
            metadatas=mm_metas,
            documents=mm_docs,
        )

        return {
            "indexed_text_items": len([m for m in mm_metas if m.get("modality") == "text"]),
            "indexed_image_items": image_count,
            "extracted_pdf_diagrams": extracted_from_pdf,
            "collection": self.cfg.multimodal_collection,
            "resume_mode": bool(resume),
            "already_indexed_text_items": len(existing_text_chunk_ids),
            "already_indexed_image_items": len(existing_image_names),
        }

    def _retrieve_context_from_embedding(
        self,
        *,
        vector: list[float],
        class_: str,
        subject: str,
        chapter: str | None,
        top_k: int,
    ) -> list[dict]:
        where_filters = [
            {"class": {"$eq": str(class_)}},
            {"subject": {"$eq": str(subject)}},
        ]
        if chapter:
            where_filters.append({"chapter": {"$eq": str(chapter)}})
        where = {"$and": where_filters}

        res = self.store.query_multimodal(
            query_embedding=vector,
            n_results=max(1, top_k),
            where=where,
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "document": str(doc or ""),
                    "metadata": dict(meta or {}),
                    "similarity": _cosine_from_distance(float(dist)),
                }
            )
        return out

    def diagram_explain(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        class_: str,
        subject: str,
        chapter: str | None,
        user_query: str | None = None,
        top_k: int = 5,
    ) -> dict:
        client = self._client()

        # 1. Extract text from the image to use for retrieval
        extract_prompt = "Extract all readable text, mathematical formulas, and questions from this image exactly as they appear. If there is no text, reply with '<NO_TEXT>'."
        ext_resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[
                extract_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config={"temperature": 0.0},
        )
        extracted_text = (getattr(ext_resp, "text", "") or "").strip()

        combined_query = user_query or ""
        if extracted_text and "<NO_TEXT>" not in extracted_text:
            combined_query += f"\n\nExtracted text from image: {extracted_text}"
        
        combined_query = combined_query.strip()

        # 2. Embed the text instead if we found any, otherwise fallback to image vector
        if combined_query:
            vector = self._embed_text(combined_query)
        else:
            vector = self._embed_bytes(blob=image_bytes, mime_type=mime_type)

        hits = self._retrieve_context_from_embedding(
            vector=vector,
            class_=class_,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )

        context_blocks: list[str] = []
        for i, h in enumerate(hits, start=1):
            m = h["metadata"]
            context_blocks.append(
                (
                    f"[Hit {i}] modality={m.get('modality')} sim={h['similarity']:.3f}\n"
                    f"chapter={m.get('chapter', '')} source={m.get('source_pdf', m.get('source_image', 'n/a'))}\n"
                    f"text={h['document']}"
                )
            )

        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant NCERT context found."
        client = self._client()
        prompt_lines = [
            "You are a friendly NCERT AI tutor. The student has uploaded an image.",
            f"The student also asked this specific question about the image: \"{user_query}\"" if user_query else "",
            "Please respond conversationally and naturally, without using rigid numbered lists to repeat internal instructions.",
            "1. Use the retrieved NCERT context as your primary guide.",
            "2. If the image contains a standard academic/exam question (e.g. maths or science problems from CBSE/NCERT), you MAY use your own mathematical and scientific formulation to solve it step-by-step.",
            "3. ONLY refuse to answer if the image represents something completely non-academic and unrelated to the NCERT syllabus. If refusing, politely say: \"I can't answer this as it is out of context of the NCERT syllabus.\"",
            "4. If answering, explain the core concepts clearly and solve any questions asked.",
            "IMPORTANT: DO NOT mention similarity scores, vector hits, or internal retrieval mechanics.\n",
            f"Retrieved NCERT context:\n{context_text}"
        ]
        prompt = "\n".join([line for line in prompt_lines if line])

        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config={"temperature": 0.2},
        )
        answer = (getattr(resp, "text", "") or "").strip()
        return {"answer": answer, "retrieved": hits}

    def search_multimodal(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        image_mime_type: str | None,
        class_: str,
        subject: str,
        chapter: str | None,
        top_k: int = 5,
    ) -> list[dict]:
        if query_text:
            vector = self._embed_text(query_text)
        elif query_image_bytes:
            vector = self._embed_bytes(
                blob=query_image_bytes,
                mime_type=image_mime_type or "image/png",
            )
        else:
            raise ValueError("Provide either query_text or query_image_bytes")

        return self._retrieve_context_from_embedding(
            vector=vector,
            class_=class_,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )

    def voice_answer(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        class_: str,
        subject: str,
        chapter: str | None,
        top_k: int = 5,
    ) -> dict:
        client = self._client()
        transcribe_prompt = (
            "Transcribe the student's spoken question exactly in English text. "
            "Return only the transcript."
        )
        tx = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[
                transcribe_prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config={"temperature": 0.0},
        )
        transcript = (getattr(tx, "text", "") or "").strip()
        if not transcript:
            transcript = "Could you repeat your question more clearly?"

        answer = self.qa_tutor.answer(
            query=transcript,
            class_=class_,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )

        # Keep response TTS-friendly for frontend/browser speech synthesis.
        speak = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=(
                "Rewrite this answer for natural spoken delivery in <= 90 words, "
                "without changing facts:\n\n"
                f"{answer}"
            ),
            config={"temperature": 0.2},
        )
        spoken_text = (getattr(speak, "text", "") or "").strip() or answer
        return {"transcript": transcript, "answer": answer, "spoken_text": spoken_text}
