import os
from pathlib import Path

# New project structure
project_name = "medi_chat"

new_structure = [
    f"{project_name}/__init__.py",
    f"{project_name}/src/prompts.py",
    f"{project_name}/src/store_index.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/helpers.py",
    f"{project_name}/logger/logger.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/exception_handler.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/notebook/testing_notebook.py",
    f"{project_name}/templates/chatbot.html",
    f"{project_name}/templates/index.html",
    f"{project_name}/static/style.css",    
    "app.py",
    "mcp_server.py",
    "requirements.txt",
    "Dockerfile",
    "demo.py",
    "setup.py",
    ".env",
]


def create_new_structure():
    for filepath in new_structure:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        if filedir:
            os.makedirs(filedir, exist_ok=True)

        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                pass
            print(f"Created file: {filepath}")
        else:
            print(f"File already exists: {filepath}")

if __name__ == "__main__":
    print("Creating new structure...")
    create_new_structure()
    print("Project structure updated.")
