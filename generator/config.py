import os

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

GEN_MODEL = os.environ.get("GEN_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_api_keys() -> list:
    """Returns the configured Groq API key(s) as a list.

    Set GROQ_API_KEYS (comma-separated) to enable round-robin failover
    across multiple keys/accounts when one hits a rate limit. Falls back to
    the single GROQ_API_KEY if GROQ_API_KEYS is not set.
    """
    keys_raw = os.environ.get("GROQ_API_KEYS", "").strip()
    if keys_raw:
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GROQ_API_KEY")
    if not single:
        raise RuntimeError(
            "No Groq API key is set. Copy .env.example to .env and set "
            "GROQ_API_KEY (single key) or GROQ_API_KEYS (comma-separated, "
            "for round-robin failover), or export one in your shell."
        )
    return [single]


def get_groq_client(api_key: str = None) -> Groq:
    if api_key is None:
        api_key = get_api_keys()[0]
    return Groq(api_key=api_key)


def create_chat_completion(**kwargs):
    """Calls chat.completions.create, rotating across all configured API
    keys on a rate-limit error so one exhausted key's quota doesn't abort
    a run. Raises the last rate-limit error if every key is exhausted.
    """
    keys = get_api_keys()
    last_error = None
    for api_key in keys:
        client = get_groq_client(api_key=api_key)
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            last_error = exc
            continue
    raise last_error
