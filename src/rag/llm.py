"""Gemini client (free tier) with retry + fallback models. Key read from .env."""
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "").split(",") if m.strip()]
RETRYABLE = {429, 500, 503}   # quota reached / server error / overloaded


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing: add it to the .env file")
    return genai.Client(api_key=key)


def generate(prompt: str, system: str | None = None, temperature: float = 0.2,
             retries: int = 2) -> tuple[str, str]:
    """Return (answer, model_used). Tries the main model, then the fallbacks."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    last_error = None
    for model in [GEMINI_MODEL, *FALLBACKS]:
        for attempt in range(retries + 1):
            try:
                r = get_client().models.generate_content(model=model, contents=prompt, config=config)
                return r.text, model
            except errors.APIError as e:
                last_error = e
                if e.code not in RETRYABLE:
                    raise                      # real error (bad key, bad request): stop
                time.sleep(2 ** attempt)       # 1s, 2s, 4s… then next attempt
    raise RuntimeError(f"All Gemini models are unavailable right now: {last_error}")


if __name__ == "__main__":
    answer, used = generate("Say 'hello, I am Valley' in Tunisian Arabic, French and English, one line each.")
    print(f"[{used}]\n{answer}")