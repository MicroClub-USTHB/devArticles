"""
╔══════════════════════════════════════════════════════════════════════╗
║                    scraper.py — Web Data Acquisition                 ║
║                                                                      ║
║  A complete, pedagogic, and production-ready web scraping toolkit.   ║
║                                                                      ║
║  Levels covered:                                                     ║
║     simple URL fetch, HTML parsing                                   ║
║     sessions, pagination, rate-limiting                              ║
║     async scraping, JS rendering, proxy rotation                     ║
║                                                                      ║
║  Dependencies (install what you need):                               ║
║    pip install requests beautifulsoup4 lxml                          ║
║    pip install httpx aiohttp                    # async              ║
║    pip install playwright                       # JS rendering       ║
║    pip install fake-useragent                   # UA rotation        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# Imports — grouped by stdlib / third-party / optional
import time
import random
import logging
import hashlib
import csv
import json
import re
import urllib.robotparser
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional
from urllib.parse import urljoin, urlparse, urlencode

import requests
from bs4 import BeautifulSoup

# Logging — always good practice, even for small scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")

# Simple, self-contained helpers
def fetch_html(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch raw HTML from a URL.

    The simplest building block. Wraps requests.get() with error handling
    so your script doesn't crash on bad URLs or network issues.

    Args:
        url:     The web page to download.
        timeout: Seconds to wait before giving up (default 10).

    Returns:
        HTML as a string, or None on failure.

    Example:
        >>> html = fetch_html("https://example.com")
        >>> print(html[:200])
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()          # raises if status >= 400
        resp.encoding = resp.apparent_encoding
        return resp.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def parse_links(html: str, base_url: str = "") -> list[str]:
    """
    Extract all hyperlinks from an HTML page.

    Converts relative paths (e.g., '/about') to absolute URLs using
    the base_url. Filters out mailto:, javascript:, and empty hrefs.

    Args:
        html:     Raw HTML string.
        base_url: Base URL used to resolve relative links.

    Returns:
        Deduplicated list of absolute URLs.

    Example:
        >>> html = fetch_html("https://example.com")
        >>> links = parse_links(html, base_url="https://example.com")
    """
    soup = BeautifulSoup(html, "lxml")
    seen, links = set(), []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def parse_text(html: str) -> str:
    """
    Strip all HTML tags and return clean readable text.

    Useful when you need the content of a page without worrying
    about the DOM structure.

    Example:
        >>> text = parse_text(fetch_html("https://example.com"))
        >>> print(text[:500])
    """
    soup = BeautifulSoup(html, "lxml")
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_table(html: str, table_index: int = 0) -> list[dict]:
    """
    Parse an HTML <table> into a list of row dictionaries.

    The first <tr> row is used as column headers.

    Args:
        html:        Raw HTML string.
        table_index: Which table to extract (0 = first). Default 0.

    Returns:
        List of dicts, one per row.

    Example:
        >>> html = fetch_html("https://en.wikipedia.org/wiki/List_of_countries_by_GDP")
        >>> rows = extract_table(html, table_index=0)
        >>> print(rows[0])
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables or table_index >= len(tables):
        logger.warning(f"Table index {table_index} not found.")
        return []

    table = tables[table_index]
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows

#    Sessions, rate-limiting, pagination
# Common browser User-Agents to rotate — helps avoid basic bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


@dataclass
class ScraperConfig:
    """
    Central configuration object for the Scraper class.

    Keeping config in a dataclass (instead of scattered kwargs) makes it:
    - Easy to serialize / log / replicate runs
    - Self-documenting
    - Shareable across scraper instances

    Attributes:
        delay_range:    (min, max) seconds to wait between requests.
        max_retries:    How many times to retry a failed request.
        retry_backoff:  Seconds added per retry (exponential-ish backoff).
        respect_robots: Whether to obey robots.txt rules.
        cache_dir:      Path to cache HTML. None = no caching.
        headers:        Extra HTTP headers sent with every request.
        proxies:        Dict of proxies, e.g. {"http": "http://1.2.3.4:8080"}.
    """
    delay_range: tuple[float, float] = (1.0, 3.0)
    max_retries: int = 3
    retry_backoff: float = 2.0
    respect_robots: bool = True
    cache_dir: Optional[Path] = None
    headers: dict = field(default_factory=dict)
    proxies: Optional[dict] = None


class Scraper:
    """
    A well-mannered, configurable web scraper.

    Features:
      ✓ Polite rate-limiting (random delays between requests)
      ✓ Automatic retries with exponential backoff
      ✓ robots.txt compliance
      ✓ Optional on-disk HTML caching (saves bandwidth on repeated runs)
      ✓ Session reuse (faster than reconnecting each time)
      ✓ User-Agent rotation

    Example        scraper = Scraper()
        html = scraper.get("https://example.com")

    Example (configured):
        config = ScraperConfig(delay_range=(0.5, 1.5), cache_dir=Path(".cache"))
        scraper = Scraper(config)
        for url in my_urls:
            html = scraper.get(url)
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.proxies = self.config.proxies or {}
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        key = hashlib.md5(url.encode()).hexdigest()
        return self.config.cache_dir / f"{key}.html"

    def _load_cache(self, url: str) -> Optional[str]:
        path = self._cache_path(url)
        if path and path.exists():
            logger.debug(f"Cache hit: {url}")
            return path.read_text(encoding="utf-8")
        return None

    def _save_cache(self, url: str, html: str) -> None:
        path = self._cache_path(url)
        if path:
            path.write_text(html, encoding="utf-8")

    def _can_fetch(self, url: str) -> bool:
        """Check robots.txt. Returns True if scraping is allowed."""
        if not self.config.respect_robots:
            return True
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                rp = None  # can't read robots.txt → assume allowed
            self._robots_cache[robots_url] = rp
        rp = self._robots_cache[robots_url]
        return rp is None or rp.can_fetch("*", url)

    def _polite_delay(self) -> None:
        """Sleep a random duration to be kind to servers."""
        lo, hi = self.config.delay_range
        time.sleep(random.uniform(lo, hi))

    def _rotate_ua(self) -> None:
        ua = random.choice(USER_AGENTS)
        self.session.headers.update({"User-Agent": ua, **self.config.headers})

    # ── Public interface ──────────────────────────────────────

    def get(self, url: str) -> Optional[str]:
        """
        Fetch a URL respecting all config rules.

        This is the main method you'll call. It handles:
          - Cache lookup (skip network if cached)
          - robots.txt check
          - User-Agent rotation
          - Polite delay
          - Retry loop with backoff

        Returns:
            HTML string or None on permanent failure.
        """
        cached = self._load_cache(url)
        if cached:
            return cached

        if not self._can_fetch(url):
            logger.warning(f"robots.txt disallows: {url}")
            return None

        self._rotate_ua()
        self._polite_delay()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                html = resp.text
                self._save_cache(url, html)
                return html
            except requests.RequestException as e:
                wait = self.config.retry_backoff * attempt
                logger.warning(f"Attempt {attempt}/{self.config.max_retries} failed for {url}: {e}. "
                                f"Retrying in {wait:.1f}s…")
                time.sleep(wait)

        logger.error(f"Permanently failed: {url}")
        return None

    def scrape_urls(self, urls: list[str]) -> Generator[tuple[str, Optional[str]], None, None]:
        """
        Iterate over multiple URLs, yielding (url, html) pairs.

        Designed as a generator so you can process pages as they arrive,
        without loading everything into memory.

        Example:
            for url, html in scraper.scrape_urls(url_list):
                if html:
                    data = parse_text(html)
                    save_to_disk(data)
        """
        total = len(urls)
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{total}] Scraping: {url}")
            yield url, self.get(url)

    def crawl(
        self,
        start_url: str,
        max_pages: int = 50,
        stay_on_domain: bool = True,
        filter_fn: Optional[Callable[[str], bool]] = None,
    ) -> Generator[tuple[str, str], None, None]:
        """
        BFS crawler — follows links starting from start_url.

        Args:
            start_url:      Where to begin.
            max_pages:      Stop after visiting this many pages.
            stay_on_domain: Only follow links on the same domain.
            filter_fn:      Optional function(url) → bool to include/skip URLs.

        Yields:
            (url, html) for each successfully fetched page.

        Example:
            for url, html in scraper.crawl("https://docs.python.org", max_pages=20):
                print(url, "—", len(html), "chars")
        """
        domain = urlparse(start_url).netloc
        queue = [start_url]
        visited = set()

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            if stay_on_domain and urlparse(url).netloc != domain:
                continue
            if filter_fn and not filter_fn(url):
                continue

            html = self.get(url)
            visited.add(url)

            if html:
                yield url, html
                new_links = parse_links(html, base_url=url)
                queue.extend(l for l in new_links if l not in visited)

#   — Data extraction helpers
def extract_structured(html: str, rules: dict[str, dict]) -> dict[str, Any]:
    """
    Extract multiple fields from a page using CSS selectors.

    Instead of writing repetitive BeautifulSoup code for each field,
    define your extraction rules declaratively.

    Args:
        html:  Raw HTML.
        rules: Dict mapping field_name → {selector, attr, multiple, transform}.
               - selector:  CSS selector string.
               - attr:      HTML attribute to extract ('text' for inner text).
               - multiple:  True to get a list, False (default) for first match.
               - transform: Optional callable to post-process the value.

    Returns:
        Dict of extracted values.

    Example:
        rules = {
            "title":  {"selector": "h1", "attr": "text"},
            "price":  {"selector": ".price", "attr": "text",
                       "transform": lambda x: float(x.replace("$", ""))},
            "images": {"selector": "img", "attr": "src", "multiple": True},
        }
        data = extract_structured(html, rules)
    """
    soup = BeautifulSoup(html, "lxml")
    result = {}

    for field, rule in rules.items():
        selector = rule["selector"]
        attr = rule.get("attr", "text")
        multiple = rule.get("multiple", False)
        transform = rule.get("transform", None)

        elements = soup.select(selector)
        if not elements:
            result[field] = [] if multiple else None
            continue

        def _extract(el):
            val = el.get_text(strip=True) if attr == "text" else el.get(attr, "")
            return transform(val) if transform else val

        result[field] = [_extract(el) for el in elements] if multiple else _extract(elements[0])

    return result

#  DATA EXPORT — Save scraped data in common formats
def save_to_csv(records: list[dict], filepath: str) -> None:
    """
    Save a list of dicts to a CSV file.

    Column names are inferred from the keys of the first record.

    Example:
        save_to_csv(rows, "output/products.csv")
    """
    if not records:
        logger.warning("No records to save.")
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved {len(records)} records to {path}")


def save_to_json(data: Any, filepath: str, indent: int = 2,) -> None:
    """
    Save any JSON-serializable object to a file.

    Example:
        save_to_json({"url": "...", "data": [...]}, "output/data.json")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    logger.info(f"Saved to {path}")

#  Async scraping (httpx + asyncio)# Uncomment and use if you need to scrape hundreds of URLs fast.
# Install: pip install httpx

import asyncio
import httpx
#
async def async_fetch(url: str, client: httpx.AsyncClient) -> tuple[str, Optional[str]]:
    
     """Async version of fetch_html. Use inside an async context."""
     try:
         resp = await client.get(url, timeout=15)
         resp.raise_for_status()
         return url, resp.text
     except Exception as e:
        logger.error(f"Async fetch failed for {url}: {e}")
        return url, None
import certifi

async def scrape_many_async(urls: list[str], concurrency: int = 10) -> list[tuple]:

    """
#     Scrape many URLs concurrently.
#
#     Uses a semaphore to cap concurrent connections (be polite!).
#     Much faster than sequential for large batches.
#
#     Example:
#         results = asyncio.run(scrape_many_async(urls, concurrency=5))
#     """
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENTS[0]},verify=certifi.where() ) as client:
         async def bounded_fetch(url):
             async with semaphore:
                await asyncio.sleep(random.uniform(0.5, 1.5))
                return await async_fetch(url, client)
         return await asyncio.gather(*[bounded_fetch(u) for u in urls])

#  JavaScript rendering (Playwright)# For pages that load content via JavaScript (React, Vue, etc.)
from playwright.sync_api import sync_playwright
def fetch_js_rendered(url: str, wait_for: str = "networkidle") -> Optional[str]:
    """
    Render a JavaScript-heavy page using a real Chromium browser.    
    Args:
         url:      Page to render
         wait_for: "networkidle" | "domcontentloaded" | "load"

     Returns:
         Fully rendered HTML after JS execution.

     Example:
         html = fetch_js_rendered("https://spa-example.com/products")
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until=wait_for)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.error(f"JS rendering failed for {url}: {e}")
        return None

#  QUICK-START — Example usage when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("  scraper.py — Quick Demo")
    print("=" * 60)

    # --etch and parse one page ---
    url = "https://books.toscrape.com"
    print(f"\nmo: Fetching {url}")
    html = fetch_html(url)
    if html:
        links = parse_links(html, base_url=url)
        print(f"   Found {len(links)} links on the page")
        text = parse_text(html)
        print(f"   Page text preview: {text[:120]}…")

    # --- : configured scraper + CSV export ---
    print("\n  demo: Scraper with config")
    config = ScraperConfig(
        delay_range=(1.0, 2.0),
        cache_dir=Path(".scraper_cache"),
        max_retries=2,
        respect_robots=False,
    )
    scraper = Scraper(config)

    extraction_rules = {
        "title":  {"selector": "h3 a", "attr": "title", "multiple": True},
        "price":  {"selector": ".price_color", "attr": "text", "multiple": True},
        "rating": {"selector": ".star-rating", "attr": "class", "multiple": True},
    }

    html = scraper.get(url)
    if html:
        data = extract_structured(html, extraction_rules)
        books = [
            {"title": t, "price": p}
            for t, p in zip(data["title"][:5], data["price"][:5])
        ]
        print(f"   Sample books: {books}")
        save_to_json(books, "output/books_sample.json")
        save_to_csv(books, "output/books_sample.csv")
# --- table extraction demo ---
    wiki = "https://www.w3schools.com/html/html_tables.asp"
    html = scraper.get(wiki)
    if html:
        rows = extract_table(html, table_index=0)
        print("   First row of table:", rows[0] if rows else "No table found")
    
    print("\n   demo: Crawling 5 pages")
    for crawled_url, crawled_html in scraper.crawl(url, max_pages=5):
        print("   Crawled:", crawled_url)


    print("\n   demo: Async scraping")
    urls = ["https://books.toscrape.com"]
    results = asyncio.run(scrape_many_async(urls, concurrency=2))
    print("   Async results:", [(u, len(h) if h else None) for u, h in results])
    
    html = fetch_js_rendered("https://example.com")
    if html:
        print("   JS-rendered length:", len(html))
    
    
    
    print("\n✅ Done. Check output/ for saved files.")