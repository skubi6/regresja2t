"""
Skrypt pobiera jednolity tekst rozporządzenia Dz.U. 2025 poz. 415
(określającego obszary właściwości prokuratur) w wersji HTML,
parsuje § 3 i § 4 i zapisuje kompletną tabelę (każde miasto,
gmina lub – w przypadku m.st. Warszawy – dzielnica) wraz z właściwą
prokuraturą okręgową do pliku Excel.

Wymagane biblioteki:
    pip install requests beautifulsoup4 pandas openpyxl
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = ("https://www.infor.pl/akt-prawny/"
       "DZU.2025.091.0000415%2Crozporzadzenie-ministra-sprawiedliwosci-"
       "w-sprawie-utworzenia-wydzialow-zamiejscowych-departamentu-do-"
       "spraw-przestepczosci-zorganizowanej-i-korupcji-prokuratury-"
       "krajowej-prokuratur-regionalnych-okregow.html")

html = requests.get(URL, timeout=30).text
soup = BeautifulSoup(html, "html.parser")
text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))       # jednolita spacja

# -------------------------------------------------------------
# 1. MAPA: siedziba prokuratury rejonowej  ➜  prokuratura okręgowa
# -------------------------------------------------------------
po_map = {}
po_pattern = re.compile(
    r"Prokuraturę Okręgową w ([^–-]+?) - [^:]+?Prokuratur Rejonowych (?:w:|:) (.+?);")
for m in po_pattern.finditer(text):
    po_name = m.group(1).strip()
    seats_raw = m.group(2)
    seats = re.split(r",\s*|\si\s", seats_raw)      # separator „,” lub „ i ”
    for seat in seats:
        seat = re.sub(r".*? w ", "", seat).strip()  # „Warszawa-Mokotów w Warszawie” → „Warszawa-Mokotów”
        if seat:
            po_map[seat] = f"Prokuratura Okręgowa w {po_name}"

# -------------------------------------------------------------
# 2. PARSOWANIE § 4  – lista gmin / miast / dzielnic
# -------------------------------------------------------------
rows = []
# Wzorzec obejmuje 3 warianty: „dla miasta…”, „dla miast…”, „dla gmin…”
pr_pattern = re.compile(
    r"Prokuraturę Rejonową (?P<pr_name>.+?) dla (?P<label>miast?:|gmin:) (?P<territory>.+?)(?:;|, [0-9]+\)\s)", re.U)

for m in pr_pattern.finditer(text):
    pr_name = m.group("pr_name")
    territory_raw = m.group("territory")
    areas = [a.strip() for a in re.split(r",\s*", territory_raw)]
    # Siedziba PR to ostatnie słowo po „w ” w nazwie („…w Grodzisku Mazowieckim”)
    seat_match = re.search(r" w ([A-ZŁŚŻŹĆŃÓĄĘa-złśżźćńóąę\- ]+)", pr_name)
    seat = seat_match.group(1).strip() if seat_match else ""
    po = po_map.get(seat, "")

    for place in areas:
        if not place:          # pusta podwójna przecinkiem
            continue
        # Określamy prefix:
        prefix = "m." if m.group("label").startswith("miast") else "gm."
        # Dla dzielnic Warszawy zmieniamy „Warszawa-Mokotów” → „Mokotów” i bez prefixu
        if place.startswith("Warszawa-"):
            label = place.split("-", 1)[1]          # sama dzielnica
            prefix = ""                             # brak „m.”/„gm.”
        else:
            label = place
        full_name = f"{prefix} {label}".strip()
        rows.append({"Jednostka": full_name, "Prokuratura okręgowa": po})

# -------------------------------------------------------------
# 3. Zapis do Excela
# -------------------------------------------------------------
df = (pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values("Jednostka")
        .reset_index(drop=True))

df.to_excel("wlasciwosc_prokuratur.xlsx", index=False, engine="openpyxl")
print(f"✔  Zapisano plik: wlasciwosc_prokuratur.xlsx  ({len(df)} rekordów)")
