#!/usr/bin/env python3
"""
download_bip_okw2025.py
-----------------------
Prototype scraper that downloads *postanowienia komisarzy wyborczych*
(powołanie obwodowych komisji wyborczych) published only in municipal
BIP sites for the 2025 Polish presidential election.

USAGE EXAMPLES
==============

# 1. CSV with header "WWW" separated by semicolons
python download_bip_okw2025.py bip_sites.csv --out downloads

# 2. CSV with header "Adres WWW", comma separator
python download_bip_okw2025.py bip_sites.csv \
       --url-column "Adres WWW" --delimiter "," --workers 15

# 3. Plain text file (one URL per line)
python download_bip_okw2025.py urls.txt --plain \
       --out downloads --workers 10

Dependencies
============
pip install aiohttp aiofiles beautifulsoup4 tqdm rapidfuzz
"""

import argparse
import asyncio
import csv
import os
import re
import sys
import unicodedata
from urllib.parse import urljoin, urlparse, urlunparse, quote

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
from rapidfuzz import process, fuzz

import re

# ────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────
KEYWORDS = ("postanowienie", "komisarz", "wyborcz", "2025")
CANDIDATE_DIRS = (
    "/wybory",
    "/wybory-prezydenckie-2025",
    "/wybory-prezydenckie",
    "/wybory-prezydenta-rp-2025",
)
HEADERS = {
    "User-Agent": "BIP-scraper/1.1 (+https://github.com/your-org)"
}
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
CONNECTIONS_LIMIT = 10
# ────────────────────────────────────────────────────────────────────


def normalize_root(url: str) -> str:
    """Return scheme + netloc (without path/query)."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def is_interesting_href(href: str, text: str) -> bool:
    """True if link text+URL contains all KEYWORDS."""
    haystack = f"{href or ''} {text or ''}".lower()
    return all(k in haystack for k in KEYWORDS)


async def fetch_html(session: aiohttp.ClientSession, url: str) -> str | None:
    """Return HTML text or None."""
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as r:
            if r.status == 200 and "text/html" in r.headers.get("content-type", ""):
                return await r.text()
    except Exception:
        pass
    return None


async def download_pdf(
    session: aiohttp.ClientSession, url: str, dest: str
) -> bool:
    """Download single PDF."""
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as r:
            if r.status == 200 and "application/pdf" in r.headers.get(
                "content-type", ""
            ):
                async with aiofiles.open(dest, "wb") as f:
                    await f.write(await r.read())
                return True
    except Exception:
        pass
    return False


async def scan_site(root: str, out_dir: str, sem: asyncio.Semaphore, csv_writer):
    root = normalize_root(root)
    async with sem:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            pages: list[tuple[str, str]] = []

            # Pass 1 – brute force common sub-directories
            for path in CANDIDATE_DIRS:
                html = await fetch_html(session, root + path)
                if html:
                    pages.append((root + path, html))

            # Pass 2 – quick onsite search (many CMS-es expose /szukaj?q=…)
            search_url = root + "/szukaj?q=postanowienie+komisarz+wyborcz+2025"
            html = await fetch_html(session, search_url)
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
                    if abs_url.lower().endswith(".pdf"):
                        filename = quote(abs_url.strip("/").split("/")[-1])
                        dest_path = os.path.join(out_dir, filename)
                        if not os.path.exists(dest_path):
                            ok = await download_pdf(session, abs_url, dest_path)
                            if ok:
                                csv_writer.writerow([root, abs_url, dest_path])
                                await asyncio.sleep(0.1)  # be polite

_BRACED = re.compile(r"\{([^}|]*)\|([^}]*)\}")           # {...|...}
_HTTP   = re.compile(r"https?://[^\s,}]+")               # raw link
# ────────────────────────────────────────────────────────────────────────

def extract_url(line: str) -> str | None:
    """
    Return the URL found in one CSV row, or None if nothing matches.
    """
    # 1) preferred: {...|http://...}
    m = _BRACED.search(line)
    if m:
        url = m.group(2).strip()
        if url:                            # guard against empty group
            # normalise scheme if user wrote just 'domain.com'
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            return url

    # 2) fallback: first bare http/https in the line
    m = _HTTP.search(line)
    if m:
        return m.group(0)

    # 3) nothing found
    return None

# ────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ────────────────────────────────────────────────────────────────────
async def run_scraper(args):
    os.makedirs(args.out, exist_ok=True)
    sem = asyncio.Semaphore(args.workers)

    async with aiofiles.open(
        os.path.join(args.out, "downloads.csv"), "w", encoding="utf-8"
    ) as fh:
        # aiofiles doesn't expose csv.writer, so build manually:
        await fh.write("site,url,saved_as\n")

        async def safe_write(row):
            await fh.write(",".join(quote(c, safe='') for c in row) + "\n")

        tasks = []
        # Variant A – plain text file (one URL per line)
        if True or args.plain:
            with open(args.csv, encoding="utf-8") as src:
                for line in src:
                    url = extract_url(line.strip())
                    
                    if url:
                        tasks.append(scan_site(url, args.out, sem, safe_write))
        # Variant B – CSV file with chosen delimiter
        else:
            with open(args.csv, newline="", encoding="utf-8") as src:
                reader = csv.DictReader(src, delimiter=args.delimiter)
                for row in reader:
                    # User-specified column
                    #print ('row', row)
                    url = row.get(args.url_column) if args.url_column else None
                    #print ('got', url)
                    # Common fallbacks
                    if not url:
                        for cand in (
                            "WWW",
                            "Adres WWW",
                            "Adres strony www",
                            "Adres strony WWW (BIP)",
                        ):
                            url = row.get(cand)
                            if url:
                                break
                    # Fallback to 1st column
                    if not url and row:
                        url = next(iter(row.values()))
                    if url:
                        tasks.append(scan_site(url.strip(), args.out, sem, safe_write))

        await tqdm.gather(*tasks)


# ────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape municipal BIP sites for 2025 OKW postanowienia PDFs."
    )
    parser.add_argument(
        "csv",
        help="Input file: either CSV with at least one column of URLs or "
        "a plain text file (use --plain).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Treat input as plain text: one URL per line (no CSV parsing).",
    )
    parser.add_argument(
        "--url-column",
        help="Header name of the column that contains the URL (CSV mode).",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Field separator for CSV mode (default ';').",
    )
    parser.add_argument(
        "--out", default="downloads", help="Directory to save PDFs (default: downloads/)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Concurrent HTTP requests (default 20). Lower if you hit 429/403.",
    )

    # Python 3.10+ for the | operator; adapt if older
    asyncio.run(run_scraper(parser.parse_args()))
