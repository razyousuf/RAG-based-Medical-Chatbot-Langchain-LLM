import os
from pathlib import Path

project_name = "medi_chat"

# New clean OOP structure
new_structure = [
    f"{project_name}/__init__.py",

    # Core utilities
    f"{project_name}/src/utils/__init__.py",
    f"{project_name}/src/utils/logger.py",
    f"{project_name}/src/utils/exception.py",

    # RAG logic
    f"{project_name}/src/rag/__init__.py",
    f"{project_name}/src/rag/indexer.py",
    f"{project_name}/src/rag/helper.py",
    f"{project_name}/src/rag/prompt.py",

    # Templates & static files
    f"{project_name}/templates/chat.html",
    f"{project_name}/static/style.css",

    # Research notebooks and data
    f"{project_name}/notebook/research_notebook.ipynb",
    f"{project_name}/data/__init__.py",

    # Root-level files
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "demo.py",        
    "setup.py",
    ".env",
    "README.md",
]

def create_new_structure(files):
    for filepath in files:
        path = Path(filepath)
        parent = path.parent

        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        # Create empty file if it doesn't exist
        if not path.exists():
            path.touch()
            print(f"Created file: {path}")
        else:
            print(f"File already exists: {path}")

if __name__ == "__main__":
    print("Creating new structure...")
    create_new_structure(new_structure)
    print("Project structure updated.")
