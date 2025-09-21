import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
import time
import os
import re
from pathlib import Path
import logging
from collections import deque
import mimetypes
import hashlib
import concurrent.futures
from threading import Lock, RLock
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turbo_spider.log'),
        logging.StreamHandler()
    ]
)

class ThreadSafeTurboSpider:
    def __init__(self, base_url, download_dir="data.ddosecrets.com"):
        self.base_url = base_url.rstrip('/')
        self.base_domain = urlparse(base_url).netloc
        self.download_dir = download_dir
        
        # Thread-safe data structures
        self.visited_urls = set()
        self.visited_lock = RLock()  # Reentrant lock for visited URLs
        self.queue = deque()
        self.queue_lock = Lock()
        self.file_count = 0
        self.file_lock = Lock()
        self.page_count = 0
        self.page_lock = Lock()
        
        self.session = requests.Session()
        
        # Target file formats
        self.target_formats = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.xls', '.xlsx', '.csv'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'images': ['.jpg', '.jpeg', '.png'],
            'data': ['.json', '.xml', '.sql'],
        }
        
        # Session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        
        self.create_directory_structure()

    def create_directory_structure(self):
        """Create organized directory structure."""
        Path(self.download_dir).mkdir(exist_ok=True)
        
        for category in self.target_formats.keys():
            Path(self.download_dir, category).mkdir(exist_ok=True)
        
        Path(self.download_dir, 'pages').mkdir(exist_ok=True)
        Path(self.download_dir, 'other').mkdir(exist_ok=True)
        Path(self.download_dir, 'logs').mkdir(exist_ok=True)

    def is_same_domain(self, url):
        """Check if URL belongs to the same domain."""
        try:
            parsed = urlparse(url)
            return parsed.netloc == self.base_domain or not parsed.netloc
        except:
            return False

    def should_download(self, url):
        """Check if URL should be downloaded based on file extension."""
        if not self.is_same_domain(url):
            return False
            
        path = urlparse(url).path.lower()
        for extensions in self.target_formats.values():
            if any(path.endswith(ext) for ext in extensions):
                return True
        return False

    def normalize_url(self, url):
        """URL normalization with thread-safe caching."""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            # Sort query parameters for consistency
            params = sorted(parsed.query.split('&'))
            normalized += f"?{'&'.join(params)}"
        return normalized.rstrip('/')

    def mark_url_visited(self, url):
        """Thread-safe URL marking."""
        normalized = self.normalize_url(url)
        with self.visited_lock:
            if normalized in self.visited_urls:
                return True  # Already visited
            self.visited_urls.add(normalized)
            return False  # New URL

    def is_url_visited(self, url):
        """Thread-safe URL check."""
        normalized = self.normalize_url(url)
        with self.visited_lock:
            return normalized in self.visited_urls

    def add_to_queue(self, url):
        """Thread-safe queue addition with duplicate check."""
        normalized = self.normalize_url(url)
        with self.visited_lock:
            if normalized in self.visited_urls:
                return False
        
        with self.queue_lock:
            if normalized not in self.queue:
                self.queue.append(normalized)
                return True
        return False

    def get_from_queue(self):
        """Thread-safe queue retrieval."""
        with self.queue_lock:
            if self.queue:
                return self.queue.popleft()
        return None

    def get_queue_size(self):
        """Thread-safe queue size check."""
        with self.queue_lock:
            return len(self.queue)

    def increment_file_count(self):
        """Thread-safe file counter."""
        with self.file_lock:
            self.file_count += 1
            return self.file_count

    def increment_page_count(self):
        """Thread-safe page counter."""
        with self.page_lock:
            self.page_count += 1
            return self.page_count

    def get_filename(self, url, content_type=None):
        """Generate unique filename with extension detection."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        if '/' in path:
            filename = path.split('/')[-1] or 'index'
        else:
            filename = path or 'index'
        
        # Add extension if missing
        if '.' not in filename:
            if content_type:
                ext = mimetypes.guess_extension(content_type.split(';')[0])
                if ext:
                    filename += ext
            else:
                filename += '.html'
        
        # Sanitize and ensure uniqueness
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        if len(filename) > 150:
            name, ext = os.path.splitext(filename)
            filename = name[:150-len(ext)] + ext
        
        return filename

    def get_category(self, filename):
        """Determine file category."""
        ext = os.path.splitext(filename)[1].lower()
        for category, extensions in self.target_formats.items():
            if ext in extensions:
                return category
        return 'other'

    def download_file(self, url):
        """Thread-safe file download with duplicate prevention."""
        if self.mark_url_visited(url):
            return False  # Already processed
        
        try:
            response = self.session.get(url, timeout=15, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            filename = self.get_filename(url, content_type)
            category = self.get_category(filename)
            
            filepath = Path(self.download_dir, category, filename)
            
            # Handle filename conflicts
            counter = 1
            original_filepath = filepath
            while filepath.exists():
                name, ext = os.path.splitext(original_filepath.name)
                filepath = original_filepath.parent / f"{name}_{counter}{ext}"
                counter += 1
            
            # Download file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(filepath)
            file_count = self.increment_file_count()
            
            logging.info(f"✓ [{file_count}] {filename} ({category}) - {file_size} bytes")
            
            return True
            
        except Exception as e:
            logging.debug(f"✗ Download failed: {url} - {e}")
            return False

    def extract_links_fast(self, soup, base_url):
        """Extract links from page."""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            absolute_url = urljoin(base_url, href)
            if self.is_same_domain(absolute_url):
                links.add(absolute_url)
        
        return links

    def crawl_page(self, url):
        """Crawl a page and return new links."""
        if self.mark_url_visited(url):
            return set()  # Already visited
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                # Save HTML page
                filename = self.get_filename(url, content_type)
                filepath = Path(self.download_dir, 'pages', filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                page_count = self.increment_page_count()
                logging.info(f"📄 [{page_count}] {filename}")
                
                # Extract links
                soup = BeautifulSoup(response.text, 'html.parser')
                return self.extract_links_fast(soup, url)
            
            elif self.should_download(url):
                # Download file
                self.download_file(url)
            
            return set()
            
        except Exception as e:
            logging.debug(f"⚠️ Crawl failed: {url} - {e}")
            return set()

    def process_url(self, url):
        """Process a single URL thread-safely."""
        if self.should_download(url):
            self.download_file(url)
            return set()
        else:
            return self.crawl_page(url)

    def worker(self):
        """Worker thread function."""
        while True:
            url = self.get_from_queue()
            if url is None:
                break
            
            try:
                new_links = self.process_url(url)
                
                # Add new links to queue
                if new_links:
                    for link in new_links:
                        self.add_to_queue(link)
                        
            except Exception as e:
                logging.debug(f"Worker error: {url} - {e}")

    def get_known_urls(self):
        """Get initial URLs to crawl."""
        return [
            f"{self.base_url}/",
            f"{self.base_url}/index.html",
            f"{self.base_url}/about.html",
            f"{self.base_url}/news.html",
            f"{self.base_url}/opcw-douma/",
            f"{self.base_url}/vault7/",
            f"{self.base_url}/cablegate.html",
            f"{self.base_url}/plusd/",
            f"{self.base_url}/clinton-emails/",
            f"{self.base_url}/podesta-emails/",
            f"{self.base_url}/insurance/",
            f"{self.base_url}/static/",
            f"{self.base_url}/files/",
            f"{self.base_url}/documents/",
        ]

    def run(self, max_pages=5000, max_files=10000, max_workers=8):
        """Run the thread-safe turbo spider."""
        # Initialize queue with known URLs
        known_urls = self.get_known_urls()
        for url in known_urls:
            self.add_to_queue(url)
        
        logging.info(f"🚀 Thread-safe turbo spider starting")
        logging.info(f"📁 Directory: {self.download_dir}")
        logging.info(f"👷 Workers: {max_workers}")
        logging.info(f"🔍 Initial queue: {self.get_queue_size()} URLs")
        
        start_time = time.time()
        last_report = time.time()
        
        # Start worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial worker jobs
            futures = [executor.submit(self.worker) for _ in range(max_workers)]
            
            # Monitor progress
            while (self.get_queue_size() > 0 and 
                   self.page_count < max_pages and 
                   self.file_count < max_files):
                
                # Progress report every 3 seconds
                current_time = time.time()
                if current_time - last_report >= 3:
                    logging.info(
                        f"📊 Pages: {self.page_count}, Files: {self.file_count}, "
                        f"Queue: {self.get_queue_size()}, Visited: {len(self.visited_urls)}"
                    )
                    last_report = current_time
                
                time.sleep(0.1)
            
            # Shutdown workers
            for future in futures:
                future.cancel()
        
        # Final report
        duration = time.time() - start_time
        logging.info(f"✅ Completed in {duration:.1f}s!")
        logging.info(f"📄 Pages: {self.page_count}, 📁 Files: {self.file_count}")
        logging.info(f"🌐 Total URLs processed: {len(self.visited_urls)}")

# Thread-safe version
if __name__ == "__main__":
    try:
        spider = ThreadSafeTurboSpider(
            base_url="https://data.ddosecrets.com/",
            download_dir="data.ddosecrets.com"
        )
        
        spider.run(
            max_pages=10000,
            max_files=20000,
            max_workers=12  # More workers for faster crawling
        )
        
    except KeyboardInterrupt:
        logging.info("\n🛑 Spider interrupted by user")
    except Exception as e:
        logging.error(f"💥 Spider crashed: {e}")
        import traceback
        traceback.print_exc()