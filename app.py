import os
import uuid
import shutil
import json
import re
import subprocess
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="TDS Data Analyst Agent - Prototype")

# Get env vars once here, after loading .env
base = os.environ.get("AIPROXY_URL")
token = os.environ.get("AIPROXY_TOKEN")

print(f"AIPROXY_URL={base}")
print(f"AIPROXY_TOKEN={'SET' if token else 'NOT SET'}")

# ---------- Call LLM via AI proxy ----------
def call_llm_system(user_prompt, timeout=60):
    if not base:
        raise Exception("Set AIPROXY_URL env var")
    url = base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("WARNING: No AIPROXY_TOKEN set!")

    system_message = (
        "You are an assistant that must output ONLY a single python script (no explanation). "
        "The script will be saved as task_script.py and executed in an empty working directory that contains the uploaded files (questions.txt and any attachments). "
        "CONSTRAINTS: Use only these libraries: requests, bs4, pandas, numpy, matplotlib, io, base64, json, os, sys, re. "
        "The script MUST finish within ~2 minutes and MUST print the final answer to STDOUT as valid JSON (either an array or an object), and should not write files outside working dir. "
        "If producing an image, include it in the JSON as a data URI like 'data:image/png;base64,...' under 100KB. "
        "Do NOT call external LLMs from the script. Do not include any extraneous text — only python code."
    )

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0
    }
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---------- Extract python code ----------
def extract_python_code(text):
    m = re.search(r"```python(.*?)```", text, flags=re.S | re.I)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()

# ---------- Run script ----------
def run_script(workdir, timeout=150):
    script_path = os.path.join(workdir, "task_script.py")
    try:
        proc = subprocess.run(
            ["python", script_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"TimeoutExpired: {str(e)}"

# ---------- API endpoints with /api prefix ----------
from fastapi import APIRouter

api_router = APIRouter(prefix="/api")

@api_router.post("/")
async def analyze(files: list[UploadFile] = File(...)):
    request_id = uuid.uuid4().hex[:8]
    workdir = tempfile.mkdtemp(prefix=f"task_{request_id}_")
    try:
        uploaded_names = []
        q_text = None

        # Save uploaded files
        for f in files:
            dest = os.path.join(workdir, f.filename)
            with open(dest, "wb") as fh:
                fh.write(await f.read())
            uploaded_names.append(f.filename)
            if f.filename.lower() == "questions.txt":
                with open(dest, "r", encoding="utf-8", errors="ignore") as r:
                    q_text = r.read()

        if not q_text:
            raise HTTPException(status_code=400, detail="questions.txt is required")

        # Build prompt
        user_prompt = (
            "Here is the contents of questions.txt:\n\n"
            f"{q_text}\n\n"
            "Files uploaded: " + ", ".join(uploaded_names) + "\n\n"
            "Write a single python script (no explanation) that reads the files in the current directory and prints the final result as JSON."
        )

        # Get script from LLM
        code = extract_python_code(call_llm_system(user_prompt))

        # Attempt-run loop
        for attempt in range(1, 4):
            with open(os.path.join(workdir, "task_script.py"), "w", encoding="utf-8") as f:
                f.write(code)

            rc, sout, serr = run_script(workdir, timeout=120)

            try:
                parsed = json.loads(sout.strip() or "null")
                return JSONResponse(content=parsed)
            except Exception:
                pass

            feedback = (
                f"Attempt {attempt} failed.\nReturn code: {rc}\nSTDOUT:\n{sout}\nSTDERR:\n{serr}\n"
                "Fix the code below and return only Python code:\n" + code
            )
            new_code = extract_python_code(call_llm_system(feedback))
            if new_code.strip() == code.strip():
                break
            code = new_code

        return JSONResponse(status_code=500, content={
            "error": "Could not produce valid JSON after retries",
            "last_stdout": sout,
            "last_stderr": serr,
            "last_code": code[:4000]
        })

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

@api_router.get("/")
def root():
    return {"message": "TDS Data Analyst Agent is running"}

app.include_router(api_router)
