from prompts import TASK_PROMPT
from llm_utils import call_llm
from executor import run_python_code

def solve_tasks(questions_path: str, dataset_path: str, metadata: str):
    with open(questions_path, "r") as f:
        questions = f.read()

    prompt = TASK_PROMPT.format(
        dataset_path=dataset_path or "No dataset",
        metadata=metadata,
        questions=questions
    )
    code = call_llm(prompt)
    result = run_python_code(code)

    if result["success"]:
        return result["output"]
    else:
        return f"Error solving tasks:\n{result['error']}"
