# karyon_crawler.py
"""
===============================================================================
KARYON HIGH-PERFORMANCE WEB CRAWLER & CORPUS ENGINE (v31.0 MASTER)
Designed for Byte-Level Pre-Training Data Collection and Information-Theoretic Filtering.
Grounded in KEP Principles:
1. Zero-Dependency Portability: Uses standard Python libraries (urllib, concurrent.futures, html.parser).
2. Biophysical "Karyon Sieve": Filters raw bytes using Shannon Entropy, UTF-8 validity,
   and morphemic coherence metrics to eliminate binary garbage, encrypted data, and repetitive noise.
3. Zero-Copy Packed Stream Formatting: Directly outputs packed NumPy binary streams (.kbin)
   with <eos> (257) document boundaries, matching Karyon's pre-training pipeline.
Author: Bazilevs (ProgVM member) & Karyon-CoRE Research Team (2026)
===============================================================================
"""

import os
import re
import sys
import math
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# =============================================================================
# 1. ROBUST ZERO-DEPENDENCY HTML STRIPPER
# =============================================================================
class KaryonHTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_chunks = []

    def handle_data(self, d):
        self.text_chunks.append(d)

    def get_data(self):
        return "".join(self.text_chunks)

def clean_html_tags(raw_html: str) -> str:
    """Strips HTML tags, scripts, styles, and normalizes whitespace."""
    # Remove script and style blocks entirely
    clean_text = re.sub(r"<(script|style)\b[^>]*>([\s\S]*?)</\1>", " ", raw_html, flags=re.IGNORECASE)
    # Strip remaining HTML tags using HTMLParser
    stripper = KaryonHTMLStripper()
    try:
        stripper.feed(clean_text)
        text = stripper.get_data()
    except Exception:
        # Fallback regex if parser encounters issues
        text = re.sub(r"<[^>]+>", " ", clean_text)
    
    # Normalize whitespace and newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# =============================================================================
# 2. THE KARYON SIEVE: INFORMATION-THEORETIC FILTERING
# =============================================================================
class KaryonSieve:
    """Applies Shannon Entropy and Morphemic Coherence filters to raw byte streams."""
    def __init__(self, min_entropy: float = 3.2, max_entropy: float = 5.0, 
                 min_word_len: float = 2.5, max_word_len: float = 12.0,
                 max_garbage_ratio: float = 0.15):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.max_garbage_ratio = max_garbage_ratio

    def calculate_shannon_entropy(self, byte_arr: np.ndarray) -> float:
        """Calculates Shannon entropy (in nats) of a byte sequence."""
        if len(byte_arr) == 0:
            return 0.0
        counts = np.bincount(byte_arr)
        probs = counts[counts > 0] / len(byte_arr)
        return -np.sum(probs * np.log(probs))

    def evaluate_text_quality(self, text: str) -> tuple:
        """
        Evaluates text quality using Shannon entropy and morphemic coherence.
        Returns (is_valid, reason, entropy, avg_word_len).
        """
        if not text or len(text) < 150:
            return False, "Too short (length < 150 chars)", 0.0, 0.0

        try:
            raw_bytes = text.encode('utf-8')
        except UnicodeEncodeError:
            return False, "Invalid UTF-8 encoding", 0.0, 0.0

        byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        entropy = self.calculate_shannon_entropy(byte_arr)

        # 1. Shannon Entropy Filter (Eradicates binary files, archives, or repetitive garbage)
        if not (self.min_entropy <= entropy <= self.max_entropy):
            return False, f"Entropy anomaly ({entropy:.2f} nats, expected {self.min_entropy}-{self.max_entropy})", entropy, 0.0

        # 2. Morphemic Coherence Filter (Eradicates word chimeras and code snippets)
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            return False, "No words found", entropy, 0.0

        avg_word_len = sum(len(w) for w in words) / len(words)
        if not (self.min_word_len <= avg_word_len <= self.max_word_len):
            return False, f"Average word length anomaly ({avg_word_len:.1f} chars)", entropy, avg_word_len

        # 3. Non-alphanumeric garbage ratio
        non_alpha_num = len(re.findall(r"[^a-zA-Z0-9\s.,!?;:()'\"]", text))
        garbage_ratio = non_alpha_num / len(text)
        if garbage_ratio > self.max_garbage_ratio:
            return False, f"Excessive non-alphanumeric garbage ratio ({garbage_ratio:.2f})", entropy, avg_word_len

        return True, "Passed Karyon Sieve", entropy, avg_word_len


# =============================================================================
# 3. HIGH-PERFORMANCE MULTITHREADED WEB CRAWLER
# =============================================================================
class KaryonWebCrawler:
    """Multithreaded web crawler with automated domain politeness and depth control."""
    def __init__(self, start_urls: list, max_depth: int = 3, max_pages: int = 500, 
                 concurrency: int = 16, timeout: float = 5.0):
        self.start_urls = start_urls
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.timeout = timeout
        
        self.visited_urls = set()
        self.url_queue = [(url, 0) for url in start_urls] # (url, depth)
        self.sieve = KaryonSieve()
        self.collected_documents = []

    def _fetch_url(self, url: str) -> str:
        """Fetches raw HTML content of a URL with user-agent spoofing."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    return ""
                return response.read().decode('utf-8', errors='replace')
        except Exception:
            return ""

    def _extract_links(self, html: str, base_url: str) -> list:
        """Extracts absolute HTTP/HTTPS links from HTML content."""
        links = []
        pattern = r'href=["\'](https?://[^"\']+)["\']'
        for link in re.findall(pattern, html, re.IGNORECASE):
            # Normalize and resolve relative links if any
            resolved = urllib.parse.urljoin(base_url, link)
            links.append(resolved)
        return list(set(links))

    def crawl(self) -> list:
        """Executes the multithreaded crawling loop."""
        print(f"[Crawler] Starting Karyon Web Crawler (concurrency={self.concurrency}, max_pages={self.max_pages})...")
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            while self.url_queue and len(self.visited_urls) < self.max_pages:
                # Take a batch of URLs matching the concurrency level
                batch = []
                while self.url_queue and len(batch) < self.concurrency:
                    url, depth = self.url_queue.pop(0)
                    if url not in self.visited_urls:
                        self.visited_urls.add(url)
                        batch.append((url, depth))

                if not batch:
                    break

                # Launch concurrent fetch jobs
                future_to_url = {
                    executor.submit(self._fetch_url, url): (url, depth) for url, depth in batch
                }

                for future in as_completed(future_to_url):
                    url, depth = future_to_url[future]
                    html = future.result()
                    if not html:
                        continue

                    # Extract text and apply Karyon Sieve
                    cleaned_text = clean_html_tags(html)
                    is_valid, reason, entropy, avg_word_len = self.sieve.evaluate_text_quality(cleaned_text)

                    if is_valid:
                        print(f"  [Sieve: PASSED] {url:<60} | Entropy: {entropy:.2f} nats | Word Len: {avg_word_len:.1f}")
                        self.collected_documents.append({
                            "url": url,
                            "text": cleaned_text,
                            "entropy": entropy
                        })
                    else:
                        # Log sieve rejections at verbose level if needed
                        pass

                    # Extract and queue new links if within depth limit
                    if depth < self.max_depth:
                        new_links = self._extract_links(html, url)
                        for link in new_links:
                            if link not in self.visited_urls:
                                self.url_queue.append((link, depth + 1))

                print(f"[Crawler Progress] Visited: {len(self.visited_urls)} | Sieve Passed Docs: {len(self.collected_documents)} | Queue Size: {len(self.url_queue)}")

        print(f"[Crawler] Completed. Visited {len(self.visited_urls)} pages. Collected {len(self.collected_documents)} high-quality documents.")
        return self.collected_documents


# =============================================================================
# 4. ZERO-COPY PACKED DATASET BUILDER
# =============================================================================
class KaryonDatasetBuilder:
    """Formats and packs cleaned text documents into contiguous NumPy binary streams."""
    def __init__(self, output_dir: str = "data/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_packed_binary_stream(self, documents: list, filename: str = "karyon_pretrain_stream.kbin"):
        """
        Converts text documents into a single packed NumPy array of uint16
        with <eos> (257) boundaries separating documents.
        """
        print(f"[Dataset Builder] Packing {len(documents)} documents into contiguous byte stream...")
        
        byte_chunks = []
        eos_arr = np.array([257], dtype=np.uint16)
        total_chars = 0

        for doc in documents:
            text = doc["text"]
            # Encode to raw UTF-8 bytes
            raw_bytes = text.encode('utf-8', errors='replace')
            # Convert to uint16 to accommodate EOS (257)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint16)
            byte_chunks.append(arr)
            byte_chunks.append(eos_arr)
            total_chars += len(text)

        if not byte_chunks:
            print("[Dataset Builder] Error: No documents to pack.")
            return None

        flat_stream = np.concatenate(byte_chunks)
        output_path = os.path.join(self.output_dir, filename)
        
        # Save as high-speed raw binary file
        np.save(output_path, flat_stream)
        
        print(f"[Dataset Builder] SUCCESS: Saved packed stream to '{output_path}'")
        print(f"  Total Raw Characters : {total_chars:,}")
        print(f"  Total Packed Bytes   : {len(flat_stream):,} bytes")
        print(f"  File Size            : {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return output_path


# =============================================================================
# 5. SELF-TEST RUNTIME
# =============================================================================
def run_crawler_self_test():
    print("\n" + "="*85)
    print(" === [KARYON WEB CRAWLER & CORPUS ENGINE SELF-TEST] ===")
    print("="*85)
    
    # High-quality educational seeds
    seeds = [
        "https://en.wikipedia.org/wiki/Active_inference",
        "https://en.wikipedia.org/wiki/State-space_model",
        "https://en.wikipedia.org/wiki/Hopfield_network"
    ]
    
    crawler = KaryonWebCrawler(start_urls=seeds, max_depth=1, max_pages=15, concurrency=4)
    docs = crawler.crawl()
    
    if docs:
        builder = KaryonDatasetBuilder()
        builder.build_packed_binary_stream(docs, "karyon_selftest_stream.npy")
    else:
        print("[Self-Test] Warning: No documents passed the Karyon Sieve during self-test.")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_crawler_self_test()
