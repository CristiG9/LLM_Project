# main.py
# pip install chromadb openai python-dotenv

import os
import json
from typing import Dict, Tuple, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

from vectorDB import build_context_from_results, build_chroma_payload
from tools import parse_json, make_get_summary_tool
from LLM_Aditional import speak_text,generate_cover_image
# ---------------------- Config ----------------------
MODEL = "gpt-4.1-nano"          # Responses API model
JSONL_PATH = "book_summaries.jsonl"
COLLECTION_NAME = "book_summaries"
TOP_K = 1

BooksDict = Dict[str, Tuple[Optional[str], List[str]]]

# ---------------------- LLM instructions & tool schema ----------------------
RECOMMEND_INSTRUCTIONS = """You are a RAG book recommender.

Rules (follow strictly):
1) Use ONLY the provided CONTEXT. Do not invent books or use outside knowledge.
2) Recommend exactly ONE title that appears in the CONTEXT.
3) If no suitable title is present, return {"status":"no_match"} and nothing else.
4) Output VALID JSON ONLY:
   {
     "status": "ok" | "no_match" | "refuse",
     "title": "<exact title from CONTEXT>",
     "blurb": "<4-5 engaging sentences tailored to the user>",
     "reasons": ["<short reason 1>", "<short reason 2>"],
     "verbal": "Because you want <summarized user preference>, I recommend <Title>. <1–2 sentence why this fits, based on CONTEXT>."
   }
5) If the user request is unsafe/clearly disallowed, return {"status":"refuse"}.
6) Respond in the SAME LANGUAGE as the user query.
"""

# Flat tool schema for Responses API
SUMMARY_TOOL = {
    "type": "function",
    "name": "get_summary_by_title",
    "description": "Return the short summary for the exact book title selected from CONTEXT.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Exact title from CONTEXT"}
        },
        "required": ["title"]
    }
}

# ---------------------- Helpers: printing & media ----------------------
def print_result(data: dict, books: BooksDict):
    status = data.get("status")
    if status == "ok":
        print("\nBot>", data.get("verbal") or data.get("blurb", ""))
        title = data.get("title")
        if title and title in books:
            summary = books[title][0] or ""
            if summary:
                print("\n[Summary]\n" + summary)
    elif status == "no_match":
        print("\nBot> I couldn't match anything in the context.")
    elif status == "refuse":
        print("\nBot> I can’t help with that request.")
    else:
        print("\nBot>", data.get("verbal", ""))


# ---------------------- Main ----------------------
def main():
    # 0) Load env + clients
    load_dotenv()  # expects .env with OPENAI_API_KEY; optional CHROMA_OPENAI_API_KEY
    if not os.getenv("CHROMA_OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY"):
        os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    client = OpenAI()  # uses OPENAI_API_KEY

    # 1) Parse JSONL once (kept in memory)
    books = parse_json(JSONL_PATH)
    print(f"[load] Books: {len(books)}")

    # 2) Chroma: collection + ingest
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("CHROMA_OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    ids, docs, metas = build_chroma_payload(books)
    try:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    except Exception:
        pass  # likely duplicates on rerun; ok for quick iteration

    # 3) Semantic search
    user_q = input("\nTell me what you're interested in so I can recommend a book:\n")
    results = collection.query(
        query_texts=[user_q],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    context = build_context_from_results(dict(results), k=TOP_K)
    print("\nContext:\n---\n" + context + "\n---")

    # Keep the single valid title from metas (for safety)
    metas_list = (results.get("metadatas") or [[]])[0]
    valid_titles = {m.get("title") for m in metas_list if m.get("title")}
    only_title = next(iter(valid_titles)) if len(valid_titles) == 1 else None

    # 4) Prepare tool (real Python function the model will call)
    tool_func = make_get_summary_tool(books)

    # 5) Responses API — first call (force 1 tool call)
    input_items = [
        {"role": "system", "content": RECOMMEND_INSTRUCTIONS},
        {"role": "user",
         "content": f"USER_QUERY:\n{user_q}\n\nCONTEXT (use only this):\n---\n{context}\n---"},
    ]
    first = client.responses.create(
        model=MODEL,
        input=input_items,  # <-- pass the list, not str(...)
        tools=[SUMMARY_TOOL],
        tool_choice={"type": "function", "name": "get_summary_by_title"},
        temperature=0.2,
    )

    # If the model already returned final JSON (no tool call), use it
    final_data = None
    if first.output_text:
        try:
            parsed = json.loads(first.output_text)
            if isinstance(parsed, dict) and parsed.get("status"):
                final_data = parsed
        except json.JSONDecodeError:
            pass

    # Otherwise, extract the tool call from response.output
    call = None
    if final_data is None:
        dump = first.model_dump()
        for item in dump.get("output", []):
            if item.get("type") == "function_call":
                call = item
                break
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "function_call":
                        call = part
                        break
                if call:
                    break

    if final_data is None and call is None:
        print("\nRaw response:\n", first.output_text or "(no text and no tool call)")
        return

    # 6) If there was a tool call: run the local function and send the output back
    if final_data is None:
        call_id = call.get("call_id") or call.get("id")
        fn_name = call.get("name")
        args_json = call.get("arguments") or "{}"
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {}

        # Pull the real argument (title). If TOP_K=1 and it's empty, use the only title.
        title_arg = (args.get("title") or "").strip()
        if not title_arg and only_title:
            title_arg = only_title

        tool_result = ""
        if fn_name == SUMMARY_TOOL["name"]:
            tool_result = tool_func(title_arg) or ""

        # Second call: send the function_call_output (Responses API)
        final = client.responses.create(
            model=MODEL,
            previous_response_id=first.id,
            input=[{
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result,
            }],
            temperature=0.2,
        )

        try:
            final_data = json.loads(final.output_text or "{}")
        except json.JSONDecodeError:
            print("\nRaw response:\n", final.output_text or "")
            return

    # 7) Print final structured result
    print_result(final_data, books)

    # 8) Optional: Y/N image generation & TTS (only when we have a title)
    if final_data and final_data.get("status") == "ok" and final_data.get("title"):
        title = final_data["title"]
        blurb_or_summary = final_data.get("blurb") or (books.get(title, ("", []))[0] or "")
        verbal_text = final_data.get("verbal") or final_data.get("blurb") or f"Recommendation: {title}"

        # Ask for image
        yn_img = input("\nGenerate a cover-style image for this recommendation? (Y/N): ").strip().lower()
        if yn_img.startswith("y"):
            try:
                img_path = generate_cover_image(client, title, blurb_or_summary)
                print(f"[image saved] {img_path}")
            except Exception as e:
                print(f"[image error] {e}")

        # Ask for TTS
        yn_tts = input("Speak the recommendation as audio (TTS)? (Y/N): ").strip().lower()
        if yn_tts.startswith("y"):
            try:
                audio_path = speak_text(client, title, verbal_text, voice="alloy")
                print(f"[audio saved] {audio_path}")
            except Exception as e:
                print(f"[tts error] {e}")


if __name__ == "__main__":
    main()

