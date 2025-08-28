import json
from typing import Dict, Tuple, Optional, List

MODEL = "gpt-4.1-nano"          # Responses API model
JSONL_PATH = "book_summaries.jsonl"
COLLECTION_NAME = "book_summaries"
TOP_K = 1


# ---------------------- Data helpers ----------------------
BooksDict = Dict[str, Tuple[Optional[str], List[str]]]

def parse_json(path: str = JSONL_PATH) -> BooksDict:
    """Parse JSONL once at startup. Returns {title: (summary_short, [themes])}."""
    books: BooksDict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = row.get("title")
            if not title:
                continue
            summary = row.get("summary_short")
            themes = row.get("themes") or []
            if not isinstance(themes, list):
                themes = [str(themes)]
            books[title] = (summary, themes)
    return books

def make_get_summary_tool(books: BooksDict):
    """Closure over the in-memory dict: title -> summary_short."""
    def get_summary_by_title(title: str) -> str:
        entry = books.get(title)
        return (entry[0] or "") if entry else ""
    return get_summary_by_title