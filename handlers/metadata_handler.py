from prompts import METADATA_PROMPT
from llm_utils import call_llm
from executor import run_python_code

def extract_metadata(dataset_path: str):
    if not dataset_path:
        return "No dataset provided."
    prompt = METADATA_PROMPT.format(dataset_path=dataset_path)
    code = call_llm(prompt)
    result = run_python_code(code)
    if result["success"]:
        return result["output"]
    else:
        return f"Error extracting metadata:\n{result['error']}"
