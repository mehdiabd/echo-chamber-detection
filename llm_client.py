"""Shared LLM client utilities for community naming."""

import os
import requests
import ollama


# --- Organization LLM configuration (primary backend for naming queue) ---
ORG_LLM_URL = os.getenv("ORG_LLM_URL", "https://api.synappse.ir/api/gemma-3/v1/chat/completions")
ORG_LLM_MODEL = os.getenv("ORG_LLM_MODEL", "google/gemma-3-27b-it")
ORG_LLM_API_KEY = (os.getenv("ORG_LLM_API_KEY") or os.getenv("SYNAPPSE_API_KEY") or "").strip()

if ORG_LLM_API_KEY:
    print(f"[Org LLM] Using configured API endpoint: {ORG_LLM_MODEL}")
else:
    # Fallback to self-hosted org endpoint when API key is not provided.
    try:
        resp = requests.get("http://192.168.59.239:8002/v1/models", timeout=5)
        ORG_LLM_MODEL = resp.json()["data"][0]["id"]
        ORG_LLM_URL = "http://192.168.59.239:8002/v1/chat/completions"
        print(f"[Org LLM] Using self-hosted endpoint: {ORG_LLM_MODEL}")
    except Exception as e:
        ORG_LLM_MODEL = None
        ORG_LLM_URL = None
        print(f"[Org LLM] Not available -> {e}")


def call_llm(user_prompt: str, backend="org", temperature: float = 0.6, max_tokens: int = 60):
    """
    Generic LLM caller with temperature control.
    backend: "org" or "local_llama"
    temperature: creativity control (0.0 - 1.0)
    """
    if backend == "org":
        if not ORG_LLM_MODEL or not ORG_LLM_URL:
            return None
        try:
            payload = {
                "model": ORG_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "شما یک متخصص در نام‌گذاری جوامع و گروه‌های اجتماعی هستید."},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {"Content-Type": "application/json"}
            if ORG_LLM_API_KEY:
                headers["apikey"] = ORG_LLM_API_KEY
            resp = requests.post(ORG_LLM_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # defensive path for different response shapes
            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            if isinstance(data, dict) and "result" in data:
                return data["result"].strip()
            return None
        except Exception as e:
            print(f"[Org LLM] Failed -> {e}")
            return None

    if backend == "local_llama":
        try:
            response = ollama.chat(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": "تو یک متخصص در نام‌گذاری جوامع و گروه‌های اجتماعی هستی."},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[Local Llama] Failed -> {e}")
            return None

    raise ValueError(f"Unknown backend: {backend}")


def call_local_llama(user_prompt: str, temperature: float = 0.5):
    return call_llm(user_prompt, backend="local_llama", temperature=temperature, max_tokens=60)


def call_org_llm(user_prompt: str, temperature: float = 0.75):
    return call_llm(user_prompt, backend="org", temperature=temperature, max_tokens=80)
