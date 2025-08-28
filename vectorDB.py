from typing import Dict, Tuple, Optional, List

BooksDict = Dict[str, Tuple[Optional[str], List[str]]]

def build_chroma_payload(books: BooksDict):
    """Make (ids, docs, metas) for collection.add(...)."""
    ids, docs, metas = [], [], []
    for title, (summary, themes) in books.items():
        ids.append(title)
        docs.append(summary or "")
        metas.append({"title": title, "themes": ", ".join(themes)})
    return ids, docs, metas


def build_context_from_results(results: dict, k: int = 1) -> str:
    """Build the CONTEXT block the model will use (Title + Summary per hit)."""
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    parts = []
    for d, m in list(zip(docs, metas))[:k]:
        title = m.get("title", "Unknown")
        parts.append(f"Title: {title}\nSummary: {d}")
    return "\n\n".join(parts)
