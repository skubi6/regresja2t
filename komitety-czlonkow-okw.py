#!/usr/bin/env python3
"""
Pobiera i parsuje składy OKW (wybory prezydenckie 2025)
ze wszystkich delegatur KBW.

• zwraca pandas.DataFrame z kolumnami:
  pdf, gmina, komisja_nr, komisja_adres, czlonek, komitet
• opcjonalnie zapisuje do Excel / CSV / SQLite
"""

from __future__ import annotations

import warnings, logging
warnings.filterwarnings("ignore", message="CropBox missing")
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)       # nazwa w nowszych wersjach
logging.getLogger("pdfplumber").setLevel(logging.ERROR)  # nadgorliwe logi pdfplumber
logging.disable(logging.CRITICAL) 

import re, time, sqlite3, requests, pdfplumber, pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
# ─────────── konfiguracja ────────────
#DELEGATURY = [
#    "jelenia-gora","legnica","walbrzych","wroclaw","bydgoszcz","torun",
#    "chelm","lublin","zamosc","gorzow","zielona-gora","kalisz","lodz",
#    "piotrkow","sieradz","krakow","nowy-sacz","plock","radom","siedlce",
#    "warszawa","opole","przemysl","rzeszow","tarnobrzeg","bialystok",
#    "lomza","suwalki","gdansk","slupsk","elblag","bielsko-biala",
#    "czestochowa","katowice","kielce","olsztyn","konin","poznan",
#    "koszalin","szczecin",
#]

DELEGATURYbase = {
    # Dolnośląskie
    "jelenia-gora", "legnica", "walbrzych", "wroclaw",
    # Kujawsko-pomorskie
    "bydgoszcz", "torun",
    # Lubelskie
    "chelm", "lublin", "zamosc",
    # Lubuskie
    "gorzow", "zielona-gora",
    # Łódzkie
    "kalisz", "lodz", "piotrkow", "sieradz",
    # Małopolskie
    "krakow", "nowy-sacz",
    # Mazowieckie
    "plock", "radom", "siedlce", "warszawa",
    # Opolskie
    "opole",
    # Podkarpackie
    "przemysl", "rzeszow", "tarnobrzeg",
    # Podlaskie
    "bialystok", "lomza", "suwalki",
    # Pomorskie
    "gdansk", "slupsk", "elblag",
    # Śląskie
    "bielsko-biala", "czestochowa", "katowice",
    # Świętokrzyskie
    "kielce",
    # Warmińsko-mazurskie
    "olsztyn",
    # Wielkopolskie
    "konin", "poznan",
    # Zachodniopomorskie
    "koszalin", "szczecin",
}
DELEGATURYfull = {
'biala-podlaska',
'bialystok',
'bielsko-biala',
'bydgoszcz',
'chelm',
'ciechanow',
'czestochowa',
'danewyborcze',
'elblag',
'gdansk',
'gorzow-wielkopolski',
'jelenia-gora',
'kalisz',
'katowice',
'kielce',
'konin',
'koszalin',
'krakow',
'krosno',
'legnica',
'leszno',
'lodz',
'lomza',
'lublin',
'nowy-sacz',
'olsztyn',
'opole',
'ostroleka',
'pila',
'piotrkow-trybunalski',
'plock',
'poznan',
'przemysl',
'radom',
'rzeszow',
'siedlce',
'sieradz',
'skierniewice',
'slupsk',
'suwalki',
'szczecin',
'tarnobrzeg',
'tarnow',
'torun',
'walbrzych',
'warszawa',
'wloclawek',
'wroclaw',
'zamosc',
'zielona-gora',
}

DELEGATURY = [x for x in DELEGATURYfull]

#DELEGATURY = ['lublin']

#HDR_GMINA_RE = re.compile(
#    r"\bw\s+(mieście|gminie)\s+([A-ZŁŚŻŹĆĄĘÓŃ][\w\s\-]*)",
#    re.I
#)
HDR_GMINA_RE = re.compile(
    r"\bw\s+(mieście|gminie)\s+([A-ZŁŚŻŹĆĄĘÓŃ][^\n,]*)",  # ← stop przy \n lub przecinku
    re.I
)

BASE = "https://{}.kbw.gov.pl/wybory-i-referenda/wybory-prezydenta-rzeczypospolitej-polskiej/wybory-prezydenta-rp-w-2025-r"
#BASE = "https://{}.kbw.gov.pl/strona-glowna/wybory-prezydenta-rzeczypospolitej-polskiej/wybory-prezydenta-rp-w-2025-r"
MINIBASE= "https://{}.kbw.gov.pl"
HEADERS = {"User-Agent":"Mozilla/5.0 (X11; Ubuntu)"}
OUT_DIR = Path("pdf_okw"); OUT_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────

# ---------- 1. rekursywne wyszukiwanie PDF-ów ----------
def list_pdfs_for_delegatura(city: str, max_depth: int = 7) -> list[str]:
    root, pdfs, seen = BASE.format(city), set(), set()
    miniroot = MINIBASE.format(city)
    print ('doing', root)
    queue = [(root, 0, False)]
    while queue:
        url, depth, preselected = queue[0]
        queue = queue[1:]
        if url in seen:
            print ('   '*depth + '   ' +'seen ', url)
            continue
        if depth > max_depth:
            print ('   '*depth + '   ' +'deep ', depth, max_depth, url)
            continue
        seen.add(url)
        try:
            print ('   '*depth + '   ' +'depth', url)
            r = requests.get(url, headers=HEADERS, timeout=30)
            if "text/html" not in r.headers.get("Content-Type",""):
                continue
            print ('   '*depth + '   ' +'found')
            soup = BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            print ('   '*depth + '   ' +'except ', url)
            continue
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            lhref = href.lower()
            lPreselected = preselected or any (
                k in lhref for k in ("postanowienie","skladach","skladzie","powol","sklad","okw"))
            
            #if lhref.endswith(".pdf") and "2025" in lhref and any(
            if lhref.endswith(".pdf") and lPreselected:
                print ('   '*depth + '   ' +'pdf   ', href)
                pdfs.add(href)
            elif href.startswith(miniroot) and not lhref.endswith('.jpg'):
                print ('   '*depth + '   ' +'append', depth+1, href)
                queue.append((href, depth+1, lPreselected))
            else:
                print ('   '*depth + '   ' +'ignore', href)
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
    re.I | re.S
)

def _clean_gmina(raw: str | None) -> str | None:
    if not raw:
        return None
    g = re.sub(r"Załącznik.*", "", raw, flags=re.I)      # wytnij śmieci
    g = re.sub(r"\s+", " ", g).strip()
    return g if re.match(r"^(m\.|gm\.)\s+[A-ZŁŚŻŹĆĄĘÓŃ]", g) else None

def parse_pdf(path: Path) -> list[dict]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(p.extract_text(x_tolerance=2) or "" for p in pdf.pages)

    # 2️⃣ – wyciągamy „m./gm. Nazwa” z pierwszych 600 znaków
    m_top = HDR_GMINA_RE.search(text[:600])
    gmina_top = None
    if m_top:
        prefix = "m." if m_top.group(1).lower().startswith("mie") else "gm."
        gmina_top = f"{prefix} {m_top.group(2).strip()}"

    matches = list(HDR_RE.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        # 3️⃣ – preferuj gminę z bloku TYLKO gdy pasuje do gmina_top
        g_block = _clean_gmina(m.group(1)) if m.group(1) else None
        if g_block and gmina_top and g_block[:4].lower() != gmina_top[:4].lower():
            g_block = None                     # inny powiat? → odrzucamy

        gmina        = g_block or gmina_top
        komisja_nr   = int(m.group(2))
        komisja_addr = m.group(3).strip().rstrip(", ")

        # fallback „z adresu” (rzadkie PDF-y bez nagłówka)
        if not gmina:
            tail = komisja_addr.split(",")[-1].strip()
            if re.fullmatch(r"[A-Za-zÀ-ž\s\-]{2,}", tail):
                gmina = f"m. {tail}"

        # ---- reszta bez zmian ----
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        segment = text[start:end].replace("\n", " ")

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


def parse_pdfO2(path: Path) -> list[dict]:
    with pdfplumber.open(path) as pdf:
        text = "\n".join(
            p.extract_text(x_tolerance=2) or "" for p in pdf.pages
        )

    # ➊  spróbuj złapać gminę z nagłówka całego dokumentu
    m_top = HDR_GMINA_RE.search(text[:600])      # wystarczy początek pliku
    gmina_top = None
    if m_top:
        prefix = "m." if m_top.group(1).lower().startswith("mie") else "gm."
        gmina_top = f"{prefix} {m_top.group(2).strip()}"

    matches = list(HDR_RE.finditer(text))
    rows = []

    for i, m in enumerate(matches):
        # ➋  jeśli HDR_RE znalazł własną gminę – bierz tę,
        #    w przeciwnym razie użyj gmina_top
        gmina = m.group(1).strip() if m.group(1) else gmina_top

        komisja_nr   = int(m.group(2))
        komisja_addr = m.group(3).strip().rstrip(", ")

        # ─ fallback „z adresu” tylko gdy gminy wciąż brak
        if not gmina:
            tail_city = komisja_addr.split(",")[-1].strip()
            if re.fullmatch(r"[A-Za-zÀ-ž\s\-]{2,}", tail_city):
                gmina = f"m. {tail_city}"

        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].replace("\n", " ")

        for mem in MEMBER_RE.finditer(segment):
            rows.append({
                "pdf":           path.name,
                "gmina":         gmina,            # ← już zawsze wypełnione
                "komisja_nr":    komisja_nr,
                "komisja_adres": komisja_addr,
                "czlonek":       " ".join(mem.group(1).split()),
                "komitet":       " ".join(mem.group(2).split()),
            })
    return rows


def parse_pdfOld(path: Path) -> list[dict]:
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
        partTwo = False
        
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
    if partTwo:
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
    else:
        return pd.DataFrame()

# ------------- CLI -------------
if __name__ == "__main__":
    df = extract_okw(
        n_workers=12,
        save_excel="OKW_2025_full.xlsx",
        save_csv  =None,
        save_sqlite=None,
        partTwo=True,
    )
    print(df.head(), "\n\nŁącznie rekordów:", len(df))

# https://olsztyn.kbw.gov.pl/strona-glowna good
# https://olsztyn.kbw.gov.pl/strona_glowna
