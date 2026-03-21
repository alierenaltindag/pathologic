"""
Temporary utility script created for one-off workspace restructuring.

PURPOSE:
This script was used to perform bulk find-and-replace updates across the codebase
when transitioning to a new project structure (e.g., standardizing data/raw, data/processed 
directories, and consolidating NAS packages under a unified search module).

ANALYTICAL RELEVANCE:
This file has NO relevance to the analytical processes, models, or data pipelines.
It was generated purely as an administrative tool for refactoring static string paths
and module imports. It can be safely ignored or deleted from the project without 
affecting any of the core functionalities or experiments.
"""

import os
import re

replacements = {
    r"\bdata\.csv\b": "data/raw/data.csv",
    r"results/prepared_data_for_xgboost\.csv": "data/processed/prepared_data_for_xgboost.csv",
    r"results/variants_for_examples\.csv": "data/processed/variants_for_examples.csv",
    r"pathologic\.nas\b": "pathologic.search.nas"
}

def process_file(filepath):
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception:
        return
    
    orig_content = content
    for pattern, rep in replacements.items():
        content = re.sub(pattern, rep, content)
    
    content = content.replace("data/raw/data/raw/data.csv", "data/raw/data.csv")

    if orig_content != content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Updated {filepath}")

for root, _, files in os.walk("."):
    if ".git" in root or "__pycache__" in root or "venv" in root or "egg-info" in root or "catboost_info" in root or ".pytest_cache" in root:
        continue
    for file in files:
        if file.endswith((".py", ".sh", ".md", ".yaml")):
            process_file(os.path.join(root, file))
