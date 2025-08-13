import os
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from utils.file_utils import save_upload
from handlers.metadata_handler import extract_metadata
from handlers.task_handler import solve_tasks

load_dotenv()
app = FastAPI(title="TDS Data Analyst Agent - Prototype")

@app.post("/api/")
async def analyze(
    questions: UploadFile = File(...),
    data: UploadFile = File(None)
):
    tmp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded files
        questions_path = save_upload(questions, tmp_dir)
        data_path = save_upload(data, tmp_dir) if data else None

        # Phase 1: Metadata extraction
        metadata = extract_metadata(data_path)

        # Phase 2: Solve tasks
        answers = solve_tasks(questions_path, data_path, metadata)

        return JSONResponse(content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir)
