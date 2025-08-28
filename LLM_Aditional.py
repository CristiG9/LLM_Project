import re
from pathlib import Path
import base64

from openai import OpenAI


def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")[:60] or "output"

def _project_root() -> Path:
    # folder where main.py lives
    return Path(__file__).resolve().parent

def generate_cover_image(client: OpenAI, title: str, blurb_or_summary: str,
                        out_dir: str | Path = "Images and TTS") -> str:
    """
    Use OpenAI Images API to create a cover-style image.
    Saves to <project>/Images/cover_<slug>.png and returns that path.
    """
    prompt = (
        f"Book cover illustration for '{title}'. "
        f"Focus on the mood and main themes: {blurb_or_summary[:800]}. "
        "No text overlay, cinematic lighting, high contrast, modern style."
    )
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
    )
    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)

    out_dir = _project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"cover_{_safe_slug(title)}.png"
    out_path.write_bytes(img_bytes)
    return str(out_path)

def speak_text(client: OpenAI, title: str, text: str, voice: str = "alloy",
               out_dir: str | Path = "Images and TTS") -> str:
    """
    Use Audio Speech API (gpt-4o-mini-tts) to synthesize MP3.
    Saves to <project>/TTS/tts_<slug>.mp3 and returns that path.
    """
    out_dir = _project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"tts_{_safe_slug(title)}.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as resp:
        resp.stream_to_file(out_path)

    return str(out_path)

