from __future__ import annotations

r"""
download_bip_okw2025_sync.py
----------------------------
Single-thread scraper to fetch *postanowienia komisarzy wyborczych*
(powołanie OKW) from municipal BIP sites – Presidential Election 2025.

USAGE
=====

    # CSV with header "WWW" and ';' delimiter
    python download_bip_okw2025_sync.py bip_sites.csv --url-column WWW

    # Plain text file: one URL per line
    python download_bip_okw2025_sync.py urls.txt --plain

Dependencies
============

    pip install requests beautifulsoup4 tqdm
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urlunparse, quote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────
KEYWORDS = ("postanowienie", "komisarz", "wyborcz", "2025")

CANDIDATE_DIRS = (
    "/wybory",
    "/wybory-prezydenckie-2025",
    "/wybory-prezydenckie",
    "/wybory-prezydenta-rp-2025",
)

HEADERS = {
    "User-Agent": "BIP-sync-scraper/1.0 (+https://github.com/your-org)"
}
TIMEOUT = 30  # seconds
# ─────────────────────────────────────────────────────────────────────


def normalize_root(url: str) -> str:
    """Return `scheme://netloc` (strip path/query)."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, "", "", "", ""))


def is_interesting_href(href: str, text: str) -> bool:
    """True if link text+URL contains *all* KEYWORDS (case-insensitive)."""
    target = (href or "") + " " + (text or "")
    target = target.lower()
    return all(k in target for k in KEYWORDS)


def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok and r.headers.get("content-type", "").startswith("text/html"):
            return r.text
    except requests.RequestException:
        pass
    return None


def download_pdf(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.ok and r.headers.get("content-type", "").startswith("application/pdf"):
            dest.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def scan_site(root_url: str, out_dir: Path, writer):
    """Sequentially process one BIP root."""
    root = normalize_root(root_url)

    pages: list[tuple[str, str]] = []

    # Pass 1 – brute-force common sub-paths
    for path in CANDIDATE_DIRS:
        html = fetch_html(root + path)
        if html:
            pages.append((root + path, html))

    # Pass 2 – very simple onsite search
    search_url = root + "/szukaj?q=postanowienie+komisarz+wyborcz+2025"
    html = fetch_html(search_url)
    if html:
        pages.append((search_url, html))

    # Parse pages
    for page_url, html in pages:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True)

            if not is_interesting_href(href, text):
                continue

            abs_url = urljoin(page_url, href)
            if not abs_url.lower().endswith(".pdf"):
                continue

            filename = quote(abs_url.strip("/").split("/")[-1])
            dest_path = out_dir / filename
            if dest_path.exists():
                continue  # already downloaded

            ok = download_pdf(abs_url, dest_path)
            if ok:
                writer.writerow([root, abs_url, str(dest_path)])
                # be polite to slow servers
                time.sleep(0.3)


# ── helper: URL extractor for messy curly-brace lines ───────────────
_BRACED = re.compile(r"\{[^}|]*\|([^}]*)\}")
_HTTP = re.compile(r"https?://[^\s,}]+")


def extract_url_field(line: str) -> Optional[str]:
    """Return first http/https URL from a raw CSV/text line."""
    m = _BRACED.search(line)
    if m:
        return m.group(1).strip()
    m = _HTTP.search(line)
    return m.group(0) if m else None


# ── main entry ───────────────────────────────────────────────────────
def iterate_input(file_path: Path, args) -> Iterable[str]:
    """Yield URLs from input file (plain or CSV)."""
    if True or args.plain:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            url = extract_url_field(line)
            if url:
                yield url
    else:
        with file_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=args.delimiter)
            for row in reader:
                url = row.get(args.url_column) if args.url_column else None
                if not url:  # fallbacks
                    for cand in ("WWW", "Adres WWW", "Adres strony www"):
                        url = row.get(cand)
                        if url:
                            break
                if not url and row:
                    url = next(iter(row.values()))
                if url:
                    yield url.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Sequential scraper for 2025 OKW postanowienia PDFs (no multithread)."
    )
    ap.add_argument("input", help="CSV or TXT with BIP URLs / lines.")
    ap.add_argument(
        "--plain", action="store_true", help="Treat input as plain text (one line per URL)."
    )
    ap.add_argument(
        "--url-column",
        help="CSV header with URL (if omitted, tries common fallbacks).",
    )
    ap.add_argument(
        "--delimiter", default=";", help="CSV delimiter (default ';')."
    )
    ap.add_argument("--out", default="down", help="Download folder (default downloads/)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv = out_dir / "downloads.csv"
    log_fh = log_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_fh)
    writer.writerow(["site", "url", "saved_as"])

    urls = list(iterate_input(Path(args.input), args))
    for url in tqdm(urls, desc="Sites"):
        try:
            print ('scanning', url)
            scan_site(url, out_dir, writer)
        except Exception as e:
            print(f"[WARN] {url}: {e}", file=sys.stderr)

    log_fh.close()
    print(f"Finished. PDFs saved to {out_dir}/; log at {log_csv}")


if __name__ == "__main__":
    main()
