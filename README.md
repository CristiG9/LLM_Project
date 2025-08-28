
# RAG Book Recommender — README

A tiny CLI app that:
1) parses a `JSONL` file of books,
2) builds a Chroma vector index,
3) does semantic search on your query,
4) lets an OpenAI model **call a local tool** (`get_summary_by_title`) to fetch the canonical summary,
5) returns a clean recommendation,
6) optionally generates a **cover-style image** and **text-to-speech** audio.

---

## 1) Project structure

```
.
├─ main.py
├─ tools.py                 # parse_json(), make_get_summary_tool(), etc.
├─ vectorDB.py              # build_chroma_payload(), build_context_from_results()
├─ book_summaries.jsonl     # your data (see format below)
├─ .env                     # OPENAI_API_KEY, CHROMA_OPENAI_API_KEY
└─ Images and TTS/          # (auto-created) generated cover images and text to speech mp3 files
```

---

## 2) Prerequisites

- **Python** 3.10+
- An **OpenAI API key** with access to:
  - `gpt-4.1-nano` (Responses API)
  - `gpt-image-1` (image generation)
  - `gpt-4o-mini-tts` (text-to-speech)
- Internet connection

---

## 3) Setup

### A) Create & activate a virtual env
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### B) Install dependencies
```bash
pip install requirements.txt
```


### C) Create `.env`
Create a text file named `.env` next to `main.py`:
```
OPENAI_API_KEY=sk-************************
CHROMA_OPENAI_API_KEY=sk-************************  # can be same key
```
> Use **KEY=VALUE** lines. Don’t use a `.env.py`.

---

## 4) Prepare your data (`book_summaries.jsonl`)
There already is a file with 15 books if you want you can add more books in this file

Each line is a **JSON object** with at least `title` and `summary_short`. `themes` is optional.

**Example (1 line):**
```json
{"title":"1984","summary_short":"A dystopian society ruled by Big Brother...","themes":["surveillance","totalitarianism"]}
```

---

## 5) Run

```bash
python main.py
```

What you’ll see:
- `[load] Books: <N>`
- It will build/refresh the Chroma collection.
- Prompt: `Tell me what you're interested in so I can recommend a book:`
- After the recommendation, you’ll be asked:
  - `Generate a cover-style image for this recommendation? (Y/N)`
  - `Speak the recommendation as audio (TTS)? (Y/N)`

Outputs:
- Images are saved in `Images and TTS/cover_<title>.png`
- Audio is saved in `Images and TTS/tts_<title>.mp3`

---

## 6) How it works (quick)

1. **Parse once**: `tools.parse_json()` loads `book_summaries.jsonl` into a dict `{title: (summary_short, [themes])}`.
2. **Chroma**: Embeds `summary_short` with OpenAI embeddings and creates/updates the `book_summaries` collection.
3. **Retrieve**: Semantic search (`TOP_K=1` by default) returns the best match.
4. **Tool-calling** (Responses API):
   - The model is instructed to pick exactly one title from the **CONTEXT**.
   - We **force** a single function call to `get_summary_by_title`.
   - We run the local Python function and return its output using `function_call_output`.
5. **Answer**: We print the model’s structured JSON (verbal recommendation + reasons) and the canonical summary.
6. **Extras**: If you choose, it uses OpenAI **Images** and **Audio (TTS)** APIs to generate a cover image and spoken audio.

---

## 7) Common issues & fixes

- **`Missing OPENAI_API_KEY / CHROMA_OPENAI_API_KEY`**  
  Ensure `.env` exists and contains:
  ```
  OPENAI_API_KEY=sk-...
  CHROMA_OPENAI_API_KEY=sk-...   # optional; falls back to OPENAI_API_KEY
  ```
- **Chroma re-ingest errors (duplicates)**  
  On re-run you may see duplicate errors; current code tolerates them with `try/except`. To persist Chroma and avoid re-embedding every run, consider:
  ```python
  from chromadb.config import Settings
  client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chroma"))
  # ...
  client.persist()
  ```
---

## 8) Safety & usage notes

- This app uses your **OpenAI API key**; usage will incur costs per model policy.
- The tool enforces “use only the provided context” to reduce hallucinations.
- If your query is unsafe/disallowed, the model returns `{"status":"refuse"}`.

---

## 9) Quick demo

```
$ python main.py
[load] Books: 14

Tell me what you're interested in so I can recommend a book:
I want a book about surveillance in a dystopia

Context:
---
Title: 1984
Summary: A dystopian novel that portrays a totalitarian society...
---

Bot> Because you want a dystopian story about surveillance and truth control, I recommend 1984. ...
Generate a cover-style image for this recommendation? (Y/N): Y
[image saved] Images and TTS/cover_1984.png
Speak the recommendation as audio (TTS)? (Y/N): Y
[audio saved] Images and TTS/tts_1984.mp3
```
