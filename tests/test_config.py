import importlib
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import RateLimitError


def _rate_limit_error():
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


def test_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("GEN_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    from generator import config
    importlib.reload(config)
    assert config.GEN_MODEL == "llama-3.3-70b-versatile"
    assert config.JUDGE_MODEL == "openai/gpt-oss-120b"
    assert config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_get_groq_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    from generator import config
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.get_groq_client()


def test_get_groq_client_returns_client_with_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    from generator import config
    client = config.get_groq_client()
    assert client is not None


def test_get_api_keys_returns_single_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "solo-key")
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    from generator import config
    assert config.get_api_keys() == ["solo-key"]


def test_get_api_keys_parses_comma_separated_list(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", " key-one, key-two ,key-three")
    from generator import config
    assert config.get_api_keys() == ["key-one", "key-two", "key-three"]


def test_get_api_keys_prefers_groq_api_keys_over_single(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "solo-key")
    monkeypatch.setenv("GROQ_API_KEYS", "key-one,key-two")
    from generator import config
    assert config.get_api_keys() == ["key-one", "key-two"]


def test_get_api_keys_raises_when_neither_set(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    from generator import config
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        config.get_api_keys()


def test_create_chat_completion_rotates_to_next_key_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "key-one,key-two")
    from generator import config

    fake_completion = MagicMock()
    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = _rate_limit_error()
    working_client = MagicMock()
    working_client.chat.completions.create.return_value = fake_completion

    clients_by_key = {"key-one": failing_client, "key-two": working_client}

    with patch("generator.config.Groq", side_effect=lambda api_key: clients_by_key[api_key]):
        result = config.create_chat_completion(model="m", messages=[])

    assert result is fake_completion
    failing_client.chat.completions.create.assert_called_once()
    working_client.chat.completions.create.assert_called_once()


def test_create_chat_completion_raises_after_all_keys_exhausted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "key-one,key-two")
    from generator import config

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = _rate_limit_error()

    with patch("generator.config.Groq", return_value=failing_client):
        with pytest.raises(RateLimitError):
            config.create_chat_completion(model="m", messages=[])

    assert failing_client.chat.completions.create.call_count == 2
