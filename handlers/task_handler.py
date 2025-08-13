# handlers/task_handler.py
import os
import requests
from executor import run_python_code
from dotenv import load_dotenv

load_dotenv()

AIPROXY_URL = os.getenv("AIPROXY_URL")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

def call_llm(prompt: str):
    """
    Sends prompt to AI Proxy and gets LLM's response.
    """
    if not AIPROXY_URL or not AIPROXY_TOKEN:
        raise RuntimeError("AIPROXY_URL and AIPROXY_TOKEN must be set")

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a Python data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(f"{AIPROXY_URL}/v1/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def solve_tasks(questions_path: str, data_path: str, metadata: dict):
    """
    Reads questions, sends them to LLM with metadata, executes code if returned.
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_text = f.read()

    # Step 1: Ask LLM for an answer + code if needed
    prompt = (
        f"Questions:\n{questions_text}\n\n"
        f"Dataset metadata:\n{metadata}\n\n"
        "If answering requires code, output it inside triple backticks (```python ... ```), "
        "otherwise just give the answer."
    )
    llm_response = call_llm(prompt)

    answers = []
    code_executed = False

    if "```python" in llm_response:
        # Extract code from LLM output
        code_block = llm_response.split("```python")[1].split("```")[0].strip()
        stdout, stderr = run_python_code(code_block)
        answers.append({"answer": llm_response, "code_output": stdout, "errors": stderr})
        code_executed = True
    else:
        answers.append({"answer": llm_response})

    return {"answers": answers, "code_executed": code_executed}
