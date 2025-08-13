"""
template.py

Create an empty project structure for the 'medi_chat' project.
This includes the OOP folder layout (src/oop/...) plus compatibility
files (src/pdf_loader.py, helpers/index_store.py, etc.).

Run:
    python template.py
"""

import os
from pathlib import Path

project_name = "medi_chat"

# Files & folders to create (keeps both the OOP layout and legacy filenames)
new_structure = [
    f"{project_name}/__init__.py",
    f"{project_name}/src/__init__.py",

    # OOP package (preferred)
    f"{project_name}/src/oop/__init__.py",
    f"{project_name}/src/oop/config.py",
    f"{project_name}/src/oop/data_loader.py",
    f"{project_name}/src/oop/processing.py",
    f"{project_name}/src/oop/embeddings.py",
    f"{project_name}/src/oop/indexer.py",
    f"{project_name}/src/oop/rag.py",
    f"{project_name}/src/oop/prompts.py",
    f"{project_name}/src/oop/exception.py",
    f"{project_name}/src/oop/logger.py",

    # Legacy / compatibility module names (so you can paste original helpers if you want)
    f"{project_name}/src/prompts.py",
    f"{project_name}/src/pdf_loader.py",
    f"{project_name}/src/huggingface_embedder.py",
    f"{project_name}/src/text_splitter.py",

    # Helpers, utilities and config
    f"{project_name}/helpers/__init__.py",
    f"{project_name}/helpers/index_store.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/exception.py",
    f"{project_name}/utils/logger.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/constants.py",

    # Data, notebooks
    f"{project_name}/data/__init__.py",
    f"{project_name}/notebook/research_notebook.ipynb",

    # Tests
    "tests/__init__.py",
    "tests/test_processing.py",
    "tests/test_integration.py",

    # Root-level app and helpers
    "app.py",      # streamlit app
    "build_index.py",        # CLI to build index
    "requirements.txt",
    "Dockerfile",
    "demo.py",
    "setup.py",
    ".env",
    "README.md",
]

# Deduplicate list while preserving order
seen = set()
final_structure = []
for p in new_structure:
    if p not in seen:
        seen.add(p)
        final_structure.append(p)

def create_new_structure(files):
    for filepath in files:
        path = Path(filepath)
        parent = path.parent

        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        # Create an empty file if it doesn't exist
        if not path.exists():
            # create an empty file
            path.touch()
            print(f"Created file: {path}")
        else:
            print(f"File already exists: {path}")

if __name__ == "__main__":
    print("Creating new structure...")
    create_new_structure(final_structure)
    print("Project structure updated.")
