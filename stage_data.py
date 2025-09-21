import os
import zipfile
import shutil

# Base folders
BASE_DIR = "./"
PROCESSED_ZIPS = os.path.join(BASE_DIR, "processed")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Extensions to move into /docs (EXCLUDING .py and .log)
SUPPORTED_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".txt", ".htm", ".html", ".eml", ".pst",
    ".rtf", ".xml", ".json", ".md", ".markdown", ".xls", ".xlsx", ".csv",
    ".ini", ".cfg", ".conf", ".config",
    ".js", ".java", ".cpp", ".c", ".h", ".css", ".php", ".rb", ".go", ".rs"
)

EXCLUDED_EXTENSIONS = (".py", ".log")

def unzip_all_recursive(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        print(f"📦 Unzipping: {zip_path}")
                        zip_ref.extractall(root)
                    os.makedirs(PROCESSED_ZIPS, exist_ok=True)
                    shutil.move(zip_path, os.path.join(PROCESSED_ZIPS, file))
                    print(f"✅ Moved zip to: {PROCESSED_ZIPS}/{file}")
                except Exception as e:
                    print(f"❌ Failed to unzip {zip_path}: {e}")

def move_supported_files_to_docs(base_dir):
    os.makedirs(DOCS_DIR, exist_ok=True)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_lower = file.lower()
            ext = os.path.splitext(file_lower)[1]

            # Skip excluded extensions
            if ext in EXCLUDED_EXTENSIONS:
                continue

            # Move only supported extensions
            if ext in SUPPORTED_EXTENSIONS:
                src = os.path.join(root, file)
                dest = os.path.join(DOCS_DIR, file)

                # Avoid moving from within /docs or /processed
                if DOCS_DIR in src or PROCESSED_ZIPS in src:
                    continue

                try:
                    shutil.move(src, dest)
                    print(f"📂 Moved: {file} → /docs")
                except Exception as e:
                    print(f"❌ Failed to move {src}: {e}")

def prepare_docs():
    unzip_all_recursive(BASE_DIR)
    move_supported_files_to_docs(BASE_DIR)

if __name__ == "__main__":
    prepare_docs()
