# handlers/metadata_handler.py
import pandas as pd
import os

def extract_metadata(file_path: str):
    """
    Extracts basic metadata from a CSV/Excel file.
    Returns a dict with summary stats, column info, and shape.
    """
    if not file_path or not os.path.exists(file_path):
        return {"error": "No dataset provided"}

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file format"}

        metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(3).to_dict(orient="records"),
            "describe": df.describe(include="all").to_dict()
        }
        return metadata
    except Exception as e:
        return {"error": str(e)}
