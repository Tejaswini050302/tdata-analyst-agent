# app.py
import os, uuid, shutil, json, re, subprocess, tempfile, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests

app = FastAPI(title="TDS Data Analyst Agent - Prototype")

# ---------- small helper: call the LLM (via AI proxy) ----------
def call_llm_system(user_prompt, timeout=60):
    """
    Sends a chat completion style request to the AI proxy.
    Expects AIPROXY_URL and AIPROXY_TOKEN set in env.
    """
    base = os.environ.get("AIPROXY_URL")
    token = os.environ.get("AIPROXY_TOKEN")
    if not base:
        raise Exception("Set AIPROXY_URL env var")
    url = base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    system_message = (
        "You are an assistant that must output ONLY a single python script (no explanation). "
        "The script will be saved as task_script.py and executed in an empty working directory that contains the uploaded files (questions.txt and any attachments). "
        "CONSTRAINTS: Use only these libraries: requests, bs4, pandas, numpy, matplotlib, io, base64, json, os, sys, re. "
        "The script MUST finish within ~2 minutes and MUST print the final answer to STDOUT as valid JSON (either an array or an object), and should not write files outside working dir. "
        "If producing an image, include it in the JSON as a data URI like 'data:image/png;base64,...' and ensure it's under 100,000 bytes. "
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
    data = r.json()
    # support both choices and result format
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # fallback try
        return json.dumps(data)

# ---------- extract python code from LLM reply ----------
def extract_python_code(text):
    # find ```python ... ``` or ``` ... ``` or raw python in reply
    m = re.search(r"```python(.*?)```", text, flags=re.S|re.I)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # if nothing, assume full text is code
    return text.strip()

# ---------- run script safely (timeout + capture) ----------
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
@app.post("/")
async def analyze(files: list[UploadFile] = File(...)):
    # require questions.txt to be present by name
    # Save uploads to a temp folder for this request
    request_id = uuid.uuid4().hex[:8]
    workdir = tempfile.mkdtemp(prefix=f"task_{request_id}_")
    try:
        uploaded_names = []
        q_text = None
        for f in files:
            # Save file
            dest = os.path.join(workdir, f.filename)
            with open(dest, "wb") as fh:
                fh.write(await f.read())
            uploaded_names.append(f.filename)
            if f.filename.lower() == "questions.txt":
                with open(dest, "r", encoding="utf-8", errors="ignore") as r:
                    q_text = r.read()

        if not q_text:
            raise HTTPException(status_code=400, detail="questions.txt is required (field name must be questions.txt)")

        user_prompt = (
            "Here is the contents of questions.txt:\n\n"
            "==== START QUESTIONS.TXT ====\n"
            f"{q_text}\n"
            "==== END QUESTIONS.TXT ====\n\n"
            "Files uploaded: " + ", ".join(uploaded_names) + "\n\n"
            "Write a single python script (no explanation) that reads the files in the current directory and prints the final result as JSON to stdout. "
            "If you need sample filenames mention them in comments but the script must work with the uploaded files. Use only allowed libs and obey constraints in the system message."
        )

        # 1) ask LLM for a python script
        llm_reply = call_llm_system(user_prompt)
        code = extract_python_code(llm_reply)

        # attempt-run loop: up to 3 tries (ask LLM to fix errors)
        last_stdout = last_stderr = ""
        for attempt in range(1, 4):
            # write code to task_script.py
            with open(os.path.join(workdir, "task_script.py"), "w", encoding="utf-8") as f:
                f.write(code)

            rc, sout, serr = run_script(workdir, timeout=120)
            last_stdout, last_stderr = sout, serr

            # first check for JSON in stdout
            try:
                parsed = json.loads(sout.strip() or "null")
                # good JSON -> return
                return JSONResponse(content=parsed)
            except Exception:
                # not valid JSON
                pass

            # if return code 0 but stdout not valid JSON, or nonzero rc: ask LLM to fix
            feedback = (
                f"Attempt {attempt} failed.\n"
                f"Return code: {rc}\n"
                f"STDOUT:\n{sout}\n\nSTDERR:\n{serr}\n\n"
                "Here is the last code (between ===). Please FIX the code and RETURN ONLY the corrected python script (no explanation, no markdown fences):\n\n"
                "===\n" + code + "\n===\n"
            )
            # ask LLM to fix code
            fix_reply = call_llm_system(feedback)
            new_code = extract_python_code(fix_reply)
            # if the LLM returns the same code, stop
            if new_code.strip() == code.strip():
                break
            code = new_code

        # after attempts: return helpful failure response
        return JSONResponse(status_code=500, content={
            "error": "Could not produce valid JSON after retries",
            "last_stdout": last_stdout,
            "last_stderr": last_stderr,
            "last_code": code[:4000]  # truncate for safety
        })

    finally:
        # clean up working directory
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass
