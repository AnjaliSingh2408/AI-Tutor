import difflib
import re

# Global vocabulary (built once)
VOCAB = set()


def build_vocab_from_chunks(chunks):
    """Extract words from indexed NCERT text."""
    for ch in chunks:
        words = re.findall(r"[a-zA-Z]+", ch.text.lower())
        VOCAB.update(words)


def normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", text.lower()).strip()


# 🔥 NEW: stronger single-word correction
def correct_word(word: str) -> str:
    if not VOCAB:
        return word

    # 1️⃣ Try normal close match (relaxed cutoff)
    match = difflib.get_close_matches(word, VOCAB, n=1, cutoff=0.45)
    if match:
        return match[0]

    # 2️⃣ Try prefix-based guess (very powerful for bad typos)
    candidates = [v for v in VOCAB if v.startswith(word[:2])]
    if candidates:
        return min(candidates, key=lambda v: abs(len(v) - len(word)))

    # 3️⃣ Return original if nothing found
    return word


def correct_query(query: str) -> str:
    q = normalize(query)

    if not VOCAB:
        return q  # fallback if vocab not built

    words = q.split()
    corrected = [correct_word(w) for w in words]

    return " ".join(corrected)