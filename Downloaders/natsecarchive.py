import requests
from bs4 import BeautifulSoup
import os
import time
import csv
from urllib.parse import urljoin

BASE_URL = "https://nsarchive.gwu.edu"
POSTINGS_URL = f"{BASE_URL}/postings/all"
CSV_LOG = "ebb_downloads.csv"

def safe_get(url):
    """Fetch a page safely with error handling."""
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        return res
    except Exception as e:
        print(f"[ERROR] Could not fetch {url}: {e}")
        return None

def collect_ebb_links():
    """Scrape all postings pages for Briefing Book links."""
    links = []
    page = 0

    while True:
        index_url = f"{POSTINGS_URL}?page={page}"
        print(f"[INFO] Fetching index page: {index_url}")
        res = safe_get(index_url)
        if not res:
            break

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("div.views-row")
        if not rows:
            print("[INFO] No more postings found, stopping.")
            break

        new_links = []
        for row in rows:
            a = row.select_one("a.full-item-rollover")
            if a and "Briefing Book" in a.get_text(strip=True):
                ebb_url = urljoin(BASE_URL, a["href"])
                if ebb_url not in links:
                    new_links.append(ebb_url)

        if not new_links:
            print(f"[INFO] No new EBB links on page {page}, moving on...")
        else:
            print(f"[INFO] Found {len(new_links)} EBB links on page {page}")
            links.extend(new_links)

        page += 1
        time.sleep(2)  # polite delay

    print(f"[INFO] Total collected EBB links: {len(links)}")
    return links

def download_pdfs_from_ebb(ebb_url, idx, total, writer):
    """Visit an EBB page, find PDFs, and download them."""
    print(f"\n[INFO] ({idx}/{total}) Processing {ebb_url}")
    ebb_res = safe_get(ebb_url)
    if not ebb_res:
        return

    ebb_soup = BeautifulSoup(ebb_res.text, "html.parser")

    # Title
    title_tag = ebb_soup.find("h1")
    ebb_title = title_tag.text.strip() if title_tag else f"EBB_{idx}"

    folder_name = ebb_title.replace(" ", "_").replace("/", "_")
    os.makedirs(folder_name, exist_ok=True)

    # PDFs
    pdf_links = [urljoin(BASE_URL, a["href"])
                 for a in ebb_soup.select("a[href$='.pdf']")]
    print(f"[INFO] Found {len(pdf_links)} PDFs in {ebb_title}")

    for pdf in pdf_links:
        filename = os.path.join(folder_name, pdf.split("/")[-1])
        if os.path.exists(filename):
            print(f"    [SKIP] Already downloaded {filename}")
            continue

        print(f"    [DOWNLOAD] {pdf}")
        try:
            with requests.get(pdf, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
        except Exception as e:
            print(f"    [ERROR] Failed to download {pdf}: {e}")

    # Write to CSV log
    writer.writerow([ebb_title, ebb_url, len(pdf_links), folder_name])

def main():
    ebb_links = collect_ebb_links()

    with open(CSV_LOG, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["EBB Title", "EBB URL", "PDFs Found", "Folder Name"])

        for i, ebb_url in enumerate(ebb_links, 1):
            download_pdfs_from_ebb(ebb_url, i, len(ebb_links), writer)
            time.sleep(2)  # polite delay

    print(f"\n[INFO] Finished. Log written to {CSV_LOG}")

if __name__ == "__main__":
    main()
