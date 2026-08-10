"""Shared LLM client utilities for community naming."""

import os

import ollama
import requests
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# Organization endpoints are tried in this order: Synappse, then self-hosted.
SELF_HOSTED_MODELS_URL = os.getenv(
    "ORG_LLM_SELF_HOSTED_MODELS_URL",
    "http://192.168.59.239:8002/v1/models",
)
SELF_HOSTED_CHAT_URL = os.getenv(
    "ORG_LLM_SELF_HOSTED_CHAT_URL",
    "http://192.168.59.239:8002/v1/chat/completions",
)
SYNAPPSE_LLM_URL = os.getenv(
    "ORG_LLM_URL",
    "https://api.synappse.ir/api/chat/v1/chat/completions",
)
ORG_LLM_API_KEY = (
    os.getenv("SYNAPPSE_API_KEY") or os.getenv("ORG_LLM_API_KEY") or ""
).strip()

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
ORG_LLM_DISCOVERY_TIMEOUT = float(os.getenv("ORG_LLM_DISCOVERY_TIMEOUT", "5"))
ORG_LLM_REQUEST_TIMEOUT = float(os.getenv("ORG_LLM_REQUEST_TIMEOUT", "30"))

SELF_HOSTED_BACKEND = "self_hosted_org"
SYNAPPSE_BACKEND = "synappse_org"
LOCAL_BACKEND = "local_llama"
NAMING_BACKEND_ORDER = (
    SYNAPPSE_BACKEND,
    SELF_HOSTED_BACKEND,
    LOCAL_BACKEND,
)

_self_hosted_model = None
_self_hosted_model_checked = False
_synappse_key_warning_printed = False


def _messages(user_prompt):
    return [
        {
            "role": "system",
            "content": (
                "شما یک متخصص در نام‌گذاری جوامع "
                "و گروه‌های اجتماعی هستید."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _extract_content(data):
    if isinstance(data, dict) and data.get("choices"):
        return data["choices"][0]["message"]["content"].strip()
    if isinstance(data, dict) and data.get("result"):
        return str(data["result"]).strip()
    return None


def _discover_self_hosted_model():
    """Discover the self-hosted model once per process."""
    global _self_hosted_model, _self_hosted_model_checked
    if _self_hosted_model_checked:
        return _self_hosted_model

    _self_hosted_model_checked = True
    try:
        response = requests.get(
            SELF_HOSTED_MODELS_URL,
            timeout=ORG_LLM_DISCOVERY_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data") if isinstance(data, dict) else None
        if not models or not isinstance(models[0], dict) or not models[0].get("id"):
            raise ValueError("model discovery response does not contain data[0].id")
        _self_hosted_model = models[0]["id"]
        print(f"[Org LLM:self-hosted] Available model: {_self_hosted_model}")
    except Exception as exc:
        _self_hosted_model = None
        print(f"[Org LLM:self-hosted] Unavailable -> {exc}")
    return _self_hosted_model


def _post_chat(url, model, user_prompt, temperature, max_tokens, headers=None):
    payload = {
        "model": model,
        "messages": _messages(user_prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        url,
        json=payload,
        headers=headers or {"Content-Type": "application/json"},
        timeout=ORG_LLM_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content = _extract_content(response.json())
    if not content:
        raise ValueError("chat response does not contain model content")
    return content


def _post_synappse_chat(user_prompt, headers):
    """Send the request shape required by the Synappse chat endpoint."""
    response = requests.post(
        SYNAPPSE_LLM_URL,
        headers=headers,
        json={
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        },
        timeout=ORG_LLM_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content = _extract_content(response.json())
    if not content:
        raise ValueError("chat response does not contain model content")
    return content


def call_self_hosted_llm(user_prompt, temperature=0.6, max_tokens=60):
    model = _discover_self_hosted_model()
    if not model:
        return None
    try:
        return _post_chat(
            SELF_HOSTED_CHAT_URL,
            model,
            user_prompt,
            temperature,
            max_tokens,
        )
    except Exception as exc:
        print(f"[Org LLM:self-hosted] Failed -> {exc}")
        return None


def call_synappse_llm(user_prompt, temperature=0.6, max_tokens=60):
    global _synappse_key_warning_printed
    if not ORG_LLM_API_KEY:
        if not _synappse_key_warning_printed:
            print("[Org LLM:Synappse] Skipped: API key is not configured")
            _synappse_key_warning_printed = True
        return None

    headers = {
        "Content-Type": "application/json",
        "apikey": ORG_LLM_API_KEY,
    }
    try:
        return _post_synappse_chat(user_prompt, headers)
    except Exception as exc:
        print(f"[Org LLM:Synappse] Failed -> {exc}")
        return None


def call_local_llama(user_prompt, temperature=0.5, max_tokens=60):
    try:
        response = ollama.chat(
            model=LOCAL_LLM_MODEL,
            messages=_messages(user_prompt),
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        print(f"[Local Llama] Failed -> {exc}")
        return None


def call_llm(user_prompt, backend="org", temperature=0.6, max_tokens=60):
    """Call one backend, or both organization backends for ``backend='org'``."""
    if backend == SELF_HOSTED_BACKEND:
        return call_self_hosted_llm(user_prompt, temperature, max_tokens)
    if backend == SYNAPPSE_BACKEND:
        return call_synappse_llm(user_prompt, temperature, max_tokens)
    if backend == LOCAL_BACKEND:
        return call_local_llama(user_prompt, temperature, max_tokens)
    if backend == "org":
        return (
            call_synappse_llm(user_prompt, temperature, max_tokens)
            or call_self_hosted_llm(user_prompt, temperature, max_tokens)
        )
    raise ValueError(f"Unknown backend: {backend}")


def call_llm_with_fallback(
    user_prompt,
    validator=None,
    org_temperature=0.2,
    local_temperature=0.1,
    max_tokens=180,
):
    """Try both organization endpoints, then local Ollama, in strict order.

    When a validator is provided, an invalid model response is treated like a
    backend failure and the next backend is attempted. The returned tuple is
    ``(validated_response, backend_name)``.
    """
    temperatures = {
        SELF_HOSTED_BACKEND: org_temperature,
        SYNAPPSE_BACKEND: org_temperature,
        LOCAL_BACKEND: local_temperature,
    }
    for backend in NAMING_BACKEND_ORDER:
        response = call_llm(
            user_prompt,
            backend=backend,
            temperature=temperatures[backend],
            max_tokens=max_tokens,
        )
        if not response:
            continue

        validated = validator(response) if validator else response
        if validated:
            print(f"[Community naming] Selected backend: {backend}")
            return validated, backend
        print(f"[Community naming] Invalid response from {backend}; trying next backend")

    return None, None


def call_org_llm(user_prompt, temperature=0.75):
    return call_llm(user_prompt, backend="org", temperature=temperature, max_tokens=80)
