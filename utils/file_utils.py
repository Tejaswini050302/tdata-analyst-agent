import os
import shutil
from fastapi import UploadFile

def save_upload(upload_file: UploadFile, dest_dir: str) -> str:
    if not upload_file:
        return None
    dest_path = os.path.join(dest_dir, upload_file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return dest_path
