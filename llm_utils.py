import os
import requests

AIPROXY_URL = os.getenv("AIPROXY_URL")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

def call_llm(prompt: str, timeout=60) -> str:
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a Python data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    resp = requests.post(f"{AIPROXY_URL}/v1/chat/completions", json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
