#!/usr/bin/env python3
"""
Pobiera i parsuje składy OKW (wybory prezydenckie 2025)
ze wszystkich delegatur KBW.

• zwraca pandas.DataFrame z kolumnami:
  pdf, gmina, komisja_nr, komisja_adres, czlonek, komitet
• opcjonalnie zapisuje do Excel / CSV / SQLite
"""

from __future__ import annotations
import re, time, sqlite3, requests, pdfplumber, pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─────────── konfiguracja ────────────
DELEGATURY = [
    "jelenia-gora","legnica","walbrzych","wroclaw","bydgoszcz","torun",
    "chelm","lublin","zamosc","gorzow","zielona-gora","kalisz","lodz",
    "piotrkow","sieradz","krakow","nowy-sacz","plock","radom","siedlce",
    "warszawa","opole","przemysl","rzeszow","tarnobrzeg","bialystok",
    "lomza","suwalki","gdansk","slupsk","elblag","bielsko-biala",
    "czestochowa","katowice","kielce","olsztyn","konin","poznan",
    "koszalin","szczecin",
]
BASE    = "https://{}.kbw.gov.pl"
HEADERS = {"User-Agent":"Mozilla/5.0 (X11; Ubuntu)"}
OUT_DIR = Path("pdf_okw"); OUT_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────

# ---------- 1. rekursywne wyszukiwanie PDF-ów ----------
def list_pdfs_for_delegatura(city: str, max_depth: int = 2) -> list[str]:
    root, pdfs, seen = BASE.format(city), set(), set()
    queue = [(root, 0)]
    while queue:
        url, depth = queue.pop()
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if "text/html" not in r.headers.get("Content-Type",""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            continue
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            lhref = href.lower()
            if lhref.endswith(".pdf") and "2025" in lhref and any(
                 k in lhref for k in ("postanowienie","powol","sklad","okw")):
                pdfs.add(href)
            elif href.startswith(root) and lhref.endswith(("/",".html",".htm")):
                queue.append((href, depth+1))
    return sorted(pdfs)

# ---------- 2. pobieranie ----------
def download(url: str, dest: Path, retries: int = 2) -> Path|None:
    if dest.exists():
        return dest
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=90); r.raise_for_status()
            dest.write_bytes(r.content); return dest
        except requests.RequestException: time.sleep(3)
    return None

# ---------- 3. parsowanie ----------
MEMBER_RE = re.compile(
    r"\d+\.\s+([A-ZŁŚŻŹĆĄĘÓŃ][^,]+?),\s+zgłoszon[ay][ae]?\s+przez\s+(.+?)(?:,|$)",
    re.DOTALL,
)
# jeden regex łapie gminę *i* nagłówek komisji
HDR_RE = re.compile(
    r"(?:(gm\.\s+[A-ZŁŚŻŹĆĄĘÓŃ][\w\s\-]*|gmina\s+[A-ZŁŚŻŹĆĄĘÓŃ][\w\s\-]*"
    r"|m\.\s+[A-ZŁŚŻŹĆĄĘÓŃ][\w\s\-]*?)\s+)?"
    r"Obwodowa Komisja Wyborcza\s+Nr\s+(\d+),\s*(.+?):",
    re.I
)

def parse_pdf(path: Path) -> list[dict]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(p.extract_text(x_tolerance=2) or "" for p in pdf.pages)

    matches = list(HDR_RE.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        gmina        = m.group(1).strip() if m.group(1) else None
        komisja_nr   = int(m.group(2))
        komisja_addr = m.group(3).strip().rstrip(", ")
        start        = m.end()
        end          = matches[i+1].start() if i+1 < len(matches) else len(text)
        segment      = text[start:end].replace("\n"," ")

        for mem in MEMBER_RE.finditer(segment):
            rows.append({
                "pdf"          : path.name,
                "gmina"        : gmina,
                "komisja_nr"   : komisja_nr,
                "komisja_adres": komisja_addr,
                "czlonek"      : " ".join(mem.group(1).split()),
                "komitet"      : " ".join(mem.group(2).split()),
            })
    return rows

# ---------- 4. API eksportowe ----------
def extract_okw(
    *,
    n_workers: int = 8,
    save_excel : str|None = "OKW_2025.xlsx",
    save_csv   : str|None = None,
    save_sqlite: str|None = None,
) -> pd.DataFrame:

    # indeksacja linków
    links=set()
    for city in tqdm(DELEGATURY, desc="Indeksuję delegatury"):
        links.update(list_pdfs_for_delegatura(city))

    # pobieranie
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        fut={ex.submit(download, url, OUT_DIR/Path(url).name):url for url in links}
        list(tqdm(as_completed(fut), total=len(fut), desc="Pobieram PDF-y"))

    # parsowanie
    rec=[]
    for pdf in tqdm(OUT_DIR.glob("*.pdf"), desc="Parsuję"):
        try:        rec.extend(parse_pdf(pdf))
        except Exception as e: print("BŁĄD:", pdf, e)

    df = pd.DataFrame(rec)

    # eksport
    if save_excel : df.to_excel (save_excel , index=False, engine="xlsxwriter")
    if save_csv   : df.to_csv   (save_csv   , index=False, encoding="utf-8-sig")
    if save_sqlite: 
        with sqlite3.connect(save_sqlite) as con:
            df.to_sql("okw", con, if_exists="replace", index=False)

    return df

# ------------- CLI -------------
if __name__ == "__main__":
    df = extract_okw(
        n_workers=12,
        save_excel="OKW_2025_full.xlsx",
        save_csv  =None,
        save_sqlite=None,
    )
    print(df.head(), "\n\nŁącznie rekordów:", len(df))
