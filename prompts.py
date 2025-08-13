METADATA_PROMPT = """
You are a data analysis assistant.
Given the dataset path, write Python code to:
1. Load the dataset using pandas.
2. Print the shape, column names, and first 5 rows.
Dataset path: {dataset_path}
"""

TASK_PROMPT = """
You are a Python data analysis assistant.
Given a dataset and the following questions, write Python code to answer them.

Dataset path: {dataset_path}
Metadata: {metadata}
Questions:
{questions}

Your code must:
- Use pandas/numpy/matplotlib only.
- Print the final answers clearly.
- Encode plots to base64 if any.
"""
