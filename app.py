import os
import io
import base64
import re
import uuid
import shutil
import tempfile
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_URL = os.environ.get("AIPROXY_URL")
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

app = FastAPI(title="Hybrid Data Analyst Agent")

# -------- LLM helper (fallback) --------
def call_llm_system(user_prompt, timeout=60):
    if not AIPROXY_URL:
        raise Exception("Set AIPROXY_URL env var")
    url = AIPROXY_URL.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if AIPROXY_TOKEN:
        headers["Authorization"] = f"Bearer {AIPROXY_TOKEN}"

    system_message = (
        "You are an assistant that outputs ONLY a single python script (no explanation). "
        "The script will run in a directory with uploaded files (questions.txt and any additional files). "
        "Use only requests, pandas, numpy, matplotlib, io, base64, json, os, sys, re, seaborn. "
        "The script must print the final result as JSON to STDOUT. "
        "If producing images, include them as 'data:image/png;base64,...' under 100KB. "
        "Do not call external LLMs or write files outside working dir."
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

def extract_python_code(text):
    m = re.search(r"```python(.*?)```", text, flags=re.S|re.I)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()

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

# -------- Analysis helpers --------
def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def correlation_answer(df):
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    corr = df[cols[0]].corr(df[cols[1]])
    return round(float(corr), 6) if not pd.isna(corr) else "NaN"

def regression_slope(df):
    from sklearn.linear_model import LinearRegression
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    df_clean = df[[cols[0], cols[1]]].dropna()
    if df_clean.empty:
        return "NaN"
    model = LinearRegression()
    X = df_clean[cols[0]].values.reshape(-1,1)
    y = df_clean[cols[1]].values
    model.fit(X, y)
    return round(float(model.coef_[0]),6)

def plot_scatter_with_regression(df):
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "No plot data"
    df_clean = df[[cols[0], cols[1]]].dropna()
    if df_clean.empty:
        return "No plot data"
    fig, ax = plt.subplots()
    sns.scatterplot(x=cols[0], y=cols[1], data=df_clean, ax=ax)
    sns.regplot(x=cols[0], y=cols[1], data=df_clean, scatter=False,
                color='red', ax=ax, line_kws={"linestyle":"dotted"})
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    uri = f"data:image/png;base64,{img_b64}"
    return uri if len(uri) < 100000 else "Plot too large"

def extract_url(text):
    m = re.search(r"https?://[^\s]+", text)
    return m.group(0) if m else None

# -------- API endpoint --------
@app.post("/api/")
async def analyze(files: list[UploadFile] = File(...)):
    workdir = tempfile.mkdtemp()
    try:
        question_text = None
        dfs = {}
        uploaded_files = []

        for f in files:
            dest_path = os.path.join(workdir, f.filename)
            with open(dest_path, "wb") as fh:
                fh.write(await f.read())
            uploaded_files.append(f.filename)
            if f.filename.lower() == "questions.txt":
                with open(dest_path,"r",encoding="utf-8",errors="ignore") as r:
                    question_text = r.read()
            elif f.filename.lower().endswith(".csv"):
                dfs[f.filename] = pd.read_csv(dest_path)

        if not question_text:
            raise HTTPException(status_code=400, detail="questions.txt is required")

        # Scrape table from URL if present
        url = extract_url(question_text)
        if url:
            try:
                tables = pd.read_html(url)
                if tables:
                    dfs["scraped_table"] = tables[0]
            except Exception:
                pass

        if not dfs:
            return JSONResponse(content=["No data found to analyze"])

        # Split questions
        questions = [q.strip() for q in re.split(r'[\n\.]+', question_text) if q.strip()]
        answers = []

        for q in questions:
            q_lower = q.lower()
            answered = False
            for df in dfs.values():
                if "correlation" in q_lower:
                    answers.append(correlation_answer(df))
                    answered = True
                    break
                elif "regression" in q_lower and "slope" in q_lower:
                    answers.append(regression_slope(df))
                    answered = True
                    break
                elif "plot" in q_lower or "scatterplot" in q_lower:
                    answers.append(plot_scatter_with_regression(df))
                    answered = True
                    break
                elif "count" in q_lower:
                    answers.append(str(len(df)))
                    answered = True
                    break
            if not answered:
                # fallback to LLM
                prompt = (
                    f"Questions:\n{question_text}\n\n"
                    "Uploaded files: " + ", ".join(uploaded_files) +
                    "\nWrite a single python script to answer these questions and print final result as JSON."
                )
                code = extract_python_code(call_llm_system(prompt))
                with open(os.path.join(workdir,"task_script.py"),"w",encoding="utf-8") as f:
                    f.write(code)
                rc, sout, serr = run_script(workdir)
                try:
                    answers.append(json.loads(sout.strip() or "null"))
                except Exception:
                    answers.append("LLM script failed")

        return JSONResponse(content=answers)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

@app.get("/")
def root():
    return {"message":"Hybrid Data Analyst Agent running"}
