# Truth Seeker

This project provides a complete workflow for downloading, staging, ingesting, and researching leaked and declassified archives (e.g., DDoSecrets, National Security Archive, WikiLeaks).

It automates:
- Downloading archives
- Preparing & staging files
- Ingesting into a vector database
- Running recursive research with Truth Seeker

---

## Requirements

### Python Dependencies

Install all dependencies with:

    pip install -r requirements.txt

### External Tools

- ocrmypdf (for OCR on scanned PDFs, optional if you use --skip-ocr)
- libpff (required by pypff to read PST files)
- Torrent client (e.g., qBittorrent, Transmission, aria2) for WikiLeaks archives
- LMStudio

---

## Quick Start

### 1. Clone the repo

    git clone https://github.com/RawdodReverend/TruthSeeker.git
    cd repo

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run downloaders

- National Security Archive (NSA EBBs)

      python natsecarchive.py

  Downloads and logs Briefing Book PDFs.

- DDoSecrets

      python ddosecrets.py

  Thread-safe spider for data.ddosecrets.com — downloads docs, archives, images, etc.

- WikiLeaks

  Add torrents (e.g., WikiLeaksTorrentArchive_archive.torrent) to your torrent client and wait for completion.  
  After the download completes, move the files into the project’s ./data folder.

---

## Stage Data

Once downloads/torrents finish, run staging:

    python stage_data.py

This will:
- Unzip all .zip archives
- Move supported files into ./docs
- Leave processed zips in ./processed

Supported extensions include:
.pdf, .doc/.docx, .txt, .eml, .pst, .json, .csv, .xls/.xlsx, .xml, .htm/.html, .rtf, .md, code files, configs, logs

---

## Ingest

Convert staged docs into vector embeddings and insert into ChromaDB:

    python ingest.py

Options:
- --skip-ocr → skip OCR processing for image-only PDFs.
  - Faster and simpler (no ocrmypdf needed).
  - Scanned PDFs without text will be skipped.

Processed docs are moved into ./processed_docs and skipped on re-runs. Failed docs go into ./failed_docs.

---

## Truth Seeker

The Truth Seeker agent lets you research across all ingested documents using recursive retrieval-augmented generation (RAG).

### Required Configuration

Before running, edit Truth_Seeker.py to point to your LM Studio instance and models:

    LM_STUDIO_API = "http://<your-host>:<port>/v1"
    EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
    CHAT_MODEL = "lmstudio-community/gemma-3-27b-it"

- Replace <your-host>:<port> with your LM Studio server.
- Ensure both the embedding model and chat model are downloaded and served.

---

### Run interactive agent

    python Truth_Seeker.py

Available modes:

- recursive: <query> → deep recursive research + generates a flowchart
- research: <query> → multi-angle research (generates queries, synthesizes evidence)
- simple: <query> → single vector search
- search: <query> → debug search results across shards
- stats → show database statistics

---

## Workflow Recap

1. Download
   - Run natsecarchive.py, ddosecrets.py, and download torrents.
   - Move completed WikiLeaks torrents into ./data.
2. Stage
   - Run stage_data.py → prepares files in ./docs.
3. Ingest
   - Run ingest.py → embed into ChromaDB (--skip-ocr optional).
4. Truth Seek
   - Configure Truth_Seeker.py with LM Studio settings.
   - Run Truth_Seeker.py → perform investigations.

---

## Notes

- OCR trade-off:
  - Default: OCR is enabled (requires ocrmypdf).
  - With --skip-ocr: faster but loses text from image-only PDFs.
- Scalability: Uses sharding across ChromaDB collections to handle large archives.
- Persistence: Data is stored in ./chroma_db by default.
