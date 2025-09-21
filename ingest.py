import os
import glob
import shutil
import hashlib
import subprocess
import tempfile
import argparse
import uuid
import email
import email.policy
from email.message import EmailMessage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import docx
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import pypff  # for PST files
import chardet
import json
import xml.etree.ElementTree as ET

from Truth_Seeker import RecursiveRAGAgent, EMBED_MODEL

# --- CONFIG ---
DOCS_FOLDER = "./docs"
PROCESSED_FOLDER = "./processed_docs"
FAILED_FOLDER = "./failed_docs"
MAX_BATCH_SIZE = 1000   # how many chunks to embed at once

# Initialize agent with sharding
agent = RecursiveRAGAgent(num_shards=10)  # Adjust number of shards as needed

# -----------------------
# FILE READERS
# -----------------------
def read_pdf(filepath, skip_ocr=False):
    text = ""
    try:
        pdf = PdfReader(filepath)
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"⚠️ Error reading PDF {filepath}: {e}")
        return ""

    if not text.strip() and not skip_ocr:
        print(f"⚡ OCRmyPDF fallback for {os.path.basename(filepath)}")
        tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        try:
            process = subprocess.Popen(
                ["ocrmypdf", "--skip-text", "--fast-web-view", "--jobs", str(os.cpu_count()), filepath, tmp_out],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            )
            process.wait()
            if process.returncode != 0:
                return ""
            pdf = PdfReader(tmp_out)
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"⚠️ OCRmyPDF error on {filepath}: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_out)
            except:
                pass
    return text


def read_doc(filepath):
    try:
        if filepath.endswith(".docx"):
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            import textract
            return textract.process(filepath).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Could not parse {filepath}: {e}")
        return ""


def read_text(filepath):
    try:
        # Try to detect encoding first
        with open(filepath, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Could not read text file {filepath}: {e}")
        return ""


def read_htm(filepath):
    """Read .htm files and extract text content"""
    try:
        with open(filepath, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"⚠️ Could not parse HTM file {filepath}: {e}")
        return ""


def read_eml(filepath):
    """Read .eml email files and extract content (excluding attachments)"""
    try:
        with open(filepath, "rb") as f:
            raw_email = f.read()
        
        # Parse the email
        msg = email.message_from_bytes(raw_email, policy=email.policy.default)
        
        text_content = []
        
        # Extract basic headers
        subject = msg.get("Subject", "")
        from_addr = msg.get("From", "")
        to_addr = msg.get("To", "")
        date = msg.get("Date", "")
        
        text_content.append(f"Subject: {subject}")
        text_content.append(f"From: {from_addr}")
        text_content.append(f"To: {to_addr}")
        text_content.append(f"Date: {date}")
        text_content.append("-" * 50)
        
        # Extract body content
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body = part.get_content()
                    if body:
                        text_content.append(body)
                elif content_type == "text/html":
                    html_body = part.get_content()
                    if html_body:
                        soup = BeautifulSoup(html_body, 'html.parser')
                        text_content.append(soup.get_text())
        else:
            # Single part message
            content_type = msg.get_content_type()
            if content_type in ["text/plain", "text/html"]:
                body = msg.get_content()
                if body:
                    if content_type == "text/html":
                        soup = BeautifulSoup(body, 'html.parser')
                        text_content.append(soup.get_text())
                    else:
                        text_content.append(body)
        
        return "\n\n".join(text_content)
    
    except Exception as e:
        print(f"⚠️ Could not parse EML file {filepath}: {e}")
        return ""


def read_pst(filepath):
    """Read .pst Outlook files and extract email content (excluding attachments)"""
    try:
        pst = pypff.file()
        pst.open(filepath)
        
        text_content = []
        
        def process_folder(folder):
            for message in folder.sub_messages:
                try:
                    # Extract basic properties
                    subject = message.subject or ""
                    sender = message.sender_name or ""
                    body = message.plain_text_body or message.html_body or ""
                    
                    if message.html_body and not message.plain_text_body:
                        # Convert HTML to text
                        soup = BeautifulSoup(body, 'html.parser')
                        body = soup.get_text()
                    
                    if any([subject, sender, body]):
                        email_text = []
                        if subject:
                            email_text.append(f"Subject: {subject}")
                        if sender:
                            email_text.append(f"From: {sender}")
                        if body:
                            email_text.append(f"Body:\n{body}")
                        
                        text_content.append("\n".join(email_text))
                        text_content.append("-" * 50)
                
                except Exception as e:
                    print(f"⚠️ Error processing message in PST: {e}")
                    continue
            
            # Process subfolders recursively
            for subfolder in folder.sub_folders:
                process_folder(subfolder)
        
        # Process root folder
        root = pst.get_root_folder()
        process_folder(root)
        
        pst.close()
        return "\n\n".join(text_content)
    
    except Exception as e:
        print(f"⚠️ Could not parse PST file {filepath}: {e}")
        return ""


def read_rtf(filepath):
    """Read RTF files"""
    try:
        from striprtf.striprtf import rtf_to_text
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)
    except ImportError:
        print(f"⚠️ Skipping RTF file {filepath} - striprtf not installed")
        return ""
    except Exception as e:
        print(f"⚠️ Could not parse RTF file {filepath}: {e}")
        return ""


def read_xml(filepath):
    """Read XML files and extract text content"""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Extract all text content from XML
        text_content = []
        
        def extract_text(element):
            if element.text and element.text.strip():
                text_content.append(element.text.strip())
            for child in element:
                extract_text(child)
            if element.tail and element.tail.strip():
                text_content.append(element.tail.strip())
        
        extract_text(root)
        return "\n".join(text_content)
    
    except Exception as e:
        print(f"⚠️ Could not parse XML file {filepath}: {e}")
        return ""


def read_json(filepath):
    """Read JSON files and convert to readable text"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        # Convert JSON to readable text format
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    except Exception as e:
        print(f"⚠️ Could not parse JSON file {filepath}: {e}")
        return ""


def read_table(filepath):
    try:
        if filepath.endswith(("xls", "xlsx")):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"⚠️ Could not parse {filepath}: {e}")
        return ""


# -----------------------
# HELPERS
# -----------------------
def move_to_failed(filepath):
    try:
        os.makedirs(FAILED_FOLDER, exist_ok=True)
        dest = os.path.join(FAILED_FOLDER, os.path.basename(filepath))
        if os.path.exists(filepath):
            shutil.move(filepath, dest)
            print(f"📂 Moved to failed_docs: {dest}")
    except Exception as e:
        print(f"⚠️ Could not move {filepath} to failed_docs: {e}")


def process_file(filepath, skip_ocr=False):
    """Read + chunk a file → return list of (text, id, metadata)."""
    ext = filepath.lower().split(".")[-1]
    
    # Route to appropriate reader based on file extension
    if ext == "pdf":
        text = read_pdf(filepath, skip_ocr=skip_ocr)
    elif ext in ["doc", "docx"]:
        text = read_doc(filepath)
    elif ext == "txt":
        text = read_text(filepath)
    elif ext == "htm":
        text = read_htm(filepath)
    elif ext == "eml":
        text = read_eml(filepath)
    elif ext == "pst":
        text = read_pst(filepath)
    elif ext == "rtf":
        text = read_rtf(filepath)
    elif ext == "xml":
        text = read_xml(filepath)
    elif ext == "json":
        text = read_json(filepath)
    elif ext in ["xls", "xlsx", "csv"]:
        text = read_table(filepath)
    elif ext in ["md", "markdown"]:
        text = read_text(filepath)  # Markdown files are just text
    elif ext in ["log", "ini", "cfg", "conf", "config"]:
        text = read_text(filepath)  # Configuration and log files
    elif ext in ["py", "js", "java", "cpp", "c", "h", "css", "html", "php", "rb", "go", "rs"]:
        text = read_text(filepath)  # Source code files
    else:
        # Try to read as text file for unknown extensions
        try:
            text = read_text(filepath)
        except:
            return []

    if not text.strip():
        move_to_failed(filepath)
        return []

    try:
        chunks = agent.smart_chunk_text(text)
    except Exception as e:
        print(f"⚠️ Chunking failed for {filepath}: {e}")
        move_to_failed(filepath)
        return []

    filename = os.path.basename(filepath)
    results = []
    for i, chunk in enumerate(chunks):
        uid = str(uuid.uuid4())  # unique ID
        metadata = {
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunk),
            "file_path": filepath,
            "file_type": ext,
        }
        results.append((chunk, uid, metadata))

    return results


# -----------------------
# INGESTION
# -----------------------
def ingest_docs_parallel(folder=DOCS_FOLDER, skip_ocr=False):
    # Collect candidate files - expanded patterns
    patterns = [
        "**/*.pdf", "**/*.docx", "**/*.doc", "**/*.txt", 
        "**/*.xls", "**/*.xlsx", "**/*.csv",
        "**/*.htm", "**/*.eml", "**/*.pst", "**/*.rtf",
        "**/*.xml", "**/*.json", "**/*.md", "**/*.markdown",
        "**/*.log", "**/*.ini", "**/*.cfg", "**/*.conf", "**/*.config",
        "**/*.py", "**/*.js", "**/*.java", "**/*.cpp", "**/*.c", "**/*.h",
        "**/*.css", "**/*.html", "**/*.php", "**/*.rb", "**/*.go", "**/*.rs"
    ]
    
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat), recursive=True))

    processed = set(os.listdir(PROCESSED_FOLDER)) if os.path.exists(PROCESSED_FOLDER) else set()
    files = [f for f in files if os.path.basename(f) not in processed]

    print(f"📥 Found {len(files)} new docs to ingest")
    
    if len(files) == 0:
        print("✅ No new documents to process")
        return

    # Phase 1: Parse + chunk in parallel
    all_chunks, all_ids, all_metas = [], [], []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_file, f, skip_ocr): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading & chunking"):
            filepath = futures[future]
            try:
                results = future.result()
                for chunk, uid, meta in results:
                    all_chunks.append(chunk)
                    all_ids.append(uid)
                    all_metas.append(meta)

                # mark file as processed (move only if it produced chunks)
                if results:
                    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                    shutil.move(filepath, os.path.join(PROCESSED_FOLDER, os.path.basename(filepath)))

            except Exception as e:
                print(f"⚠️ Worker crashed on {filepath}: {e}")
                move_to_failed(filepath)

    print(f"📦 Collected {len(all_chunks)} chunks from {len(files)} docs")

    if len(all_chunks) == 0:
        print("⚠️ No chunks were extracted from any documents")
        return

    # Phase 2: Distribute chunks to shards and bulk embed + upsert
    shard_batches = {i: {"texts": [], "ids": [], "metas": []} for i in range(agent.num_shards)}

    for chunk, uid, meta in zip(all_chunks, all_ids, all_metas):
        shard_idx = agent._get_shard_for_id(uid)
        shard_batches[shard_idx]["texts"].append(chunk)
        shard_batches[shard_idx]["ids"].append(uid)
        shard_batches[shard_idx]["metas"].append(meta)

    # Now process each shard's batch
    for shard_idx, batch in tqdm(shard_batches.items(), desc="Upserting to shards"):
        texts = batch["texts"]
        ids = batch["ids"]
        metas = batch["metas"]
        
        if not texts:
            continue
            
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:i+MAX_BATCH_SIZE]
            batch_ids = ids[i:i+MAX_BATCH_SIZE]
            batch_metas = metas[i:i+MAX_BATCH_SIZE]

            try:
                resp = agent.embed_client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
                embs = [d.embedding for d in resp.data]
                agent.shards[shard_idx].upsert(
                    documents=batch_texts,
                    embeddings=embs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
            except Exception as e:
                print(f"⚠️ Embedding batch failed for shard_{shard_idx:02d} at {i}: {e}")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest various document types into ChromaDB")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR fallback for PDFs")
    parser.add_argument("--folder", default=DOCS_FOLDER, help="Folder to scan for documents")
    args = parser.parse_args()

    print("📚 Ingesting documents into ChromaDB (optimized bulk mode with sharding)...")
    print("Supported formats: PDF, DOC/DOCX, TXT, HTM, EML, PST, RTF, XML, JSON, MD, XLS/XLSX, CSV, code files, config files, logs")
    
    ingest_docs_parallel(folder=args.folder, skip_ocr=args.skip_ocr)
    agent.get_stats()