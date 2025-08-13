from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import os, uuid, shutil, json, re, subprocess, tempfile, requests
from dotenv import load_dotenv

# Load .env
load_dotenv()
base = os.environ.get("AIPROXY_URL")
token = os.environ.get("AIPROXY_TOKEN")

print(f"AIPROXY_URL={base}")
print(f"AIPROXY_TOKEN={'SET' if token else 'NOT SET'}")

app = FastAPI(title="TDS Data Analyst Agent - Prototype")
api_router = APIRouter(prefix="/api")

# ---------- LLM call ----------
def call_llm_system(user_prompt, timeout=60):
    if not base:
        raise Exception("Set AIPROXY_URL env var")
    url = base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    system_message = (
        "You are an assistant that must output ONLY a single python script (no explanation). "
        "The script will be saved as task_script.py and executed in an empty working directory "
        "that contains the uploaded files (questions.txt and any attachments). "
        "Use only these libraries: requests, bs4, pandas, numpy, matplotlib, io, base64, json, os, sys, re. "
        "The script MUST finish within ~2 minutes and print valid JSON to STDOUT."
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

# ---------- Extract Python ----------
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

# ---------- API endpoint ----------
@api_router.post("/")
async def analyze(files: list[UploadFile] = File(...)):
    request_id = uuid.uuid4().hex[:8]
    workdir = tempfile.mkdtemp(prefix=f"task_{request_id}_")
    try:
        question_text = None
        uploaded_files = []

        for f in files:
            dest_path = os.path.join(workdir, f.filename)
            with open(dest_path, "wb") as fh:
                fh.write(await f.read())
            uploaded_files.append(f.filename)
            if f.filename.lower() == "questions.txt":
                with open(dest_path, "r", encoding="utf-8", errors="ignore") as r:
                    question_text = r.read()

        if not question_text:
            raise HTTPException(status_code=400, detail="questions.txt is required")

        # Build prompt
        prompt = (
            "Here is the content of questions.txt:\n\n"
            f"{question_text}\n\n"
            "Uploaded files: " + ", ".join(uploaded_files) + "\n\n"
            "Write a single Python script (no explanation) that reads the files in the current directory "
            "and prints the final result as valid JSON to STDOUT."
        )

        # Get initial code
        code = extract_python_code(call_llm_system(prompt))

        # Attempt-run-retry loop
        for attempt in range(1, 4):
            with open(os.path.join(workdir, "task_script.py"), "w", encoding="utf-8") as f:
                f.write(code)

            rc, stdout, stderr = run_script(workdir, timeout=120)

            try:
                parsed = json.loads(stdout.strip() or "null")
                return JSONResponse(content=parsed)
            except Exception:
                feedback = (
                    f"Attempt {attempt} failed.\nReturn code: {rc}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n"
                    "Fix the code below and return only Python code:\n" + code
                )
                new_code = extract_python_code(call_llm_system(feedback))
                if new_code.strip() == code.strip():
                    break
                code = new_code

        return JSONResponse(status_code=500, content={
            "error": "Could not produce valid JSON after retries",
            "last_stdout": stdout,
            "last_stderr": stderr,
            "last_code": code[:4000]
        })

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

@api_router.get("/")
def root():
    return {"message": "TDS Data Analyst Agent is running"}

app.include_router(api_router)
