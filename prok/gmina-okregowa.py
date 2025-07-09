"""
prokuratury_gmina_mapping.py
────────────────────────────
• Reads:  prokuratury_mapping.xlsx
• Writes:
    ├─ gmina_to_okręgowa.xlsx   (sheet “mapping” = clean result,
    │                            sheet “anomalies” = rows that break the rules)
    └─ invalid_gmina_rows.xlsx  (rows whose gmina name is malformed)

Run:  python prokuratury_gmina_mapping.py
"""

import re
import pandas as pd

import re

# -------- explicit dictionary --------
LOC_TO_NOM = {
    "Aleksandrowie Kujawskim": "Aleksandrów Kujawski",
    "Będzinie": "Będzin",
    "Białymstoku": "Białystok",
    "Biłgoraju": "Biłgoraj",
    "Biskupcu": "Biskupiec",
    "Bolesławcu": "Bolesławiec",
    "Brodnicy": "Brodnica",
    "Brzegu": "Brzeg",
    "Brzesku": "Brzesko",
    "Brzezinach": "Brzeziny",
    "Busku-Zdroju": "Busko-Zdrój",
    "Chełmie": "Chełm",
    "Chrzanowie": "Chrzanów",
    "Cieszynie": "Cieszyn",
    "Częstochowie": "Częstochowa",
    "Dąbrowie Tarnowskiej": "Dąbrowa Tarnowska",
    "Dębicy": "Dębica",
    "Działdowie": "Działdowo",
    "Garwolinie": "Garwolin",
    "Głogowie": "Głogów",
    "Goleniowie": "Goleniów",
    "Gorlicach": "Gorlice",
    "Gorzowie Wielkopolskim": "Gorzów Wielkopolski",
    "Gostyninie": "Gostynin",
    "Gostyniu": "Gostyń",
    "Grajewie": "Grajewo",
    "Grójcu": "Grójec",
    "Janowie Lubelskim": "Janów Lubelski",
    "Jarocinie": "Jarocin",
    "Jarosławiu": "Jarosław",
    "Jaśle": "Jasło",
    "Jędrzejowie": "Jędrzejów",
    "Kaliszu": "Kalisz",
    "Kartuzach": "Kartuzy",
    "Kępnie": "Kępno",
    "Kłodzku": "Kłodzko",
    "Kluczborku": "Kluczbork",
    "Kole": "Koło",
    "Kolnie": "Kolno",
    "Koninie": "Konin",
    "Krakowie": "Kraków",
    "Krośnie Odrzańskim": "Krosno Odrzańskie",
    "Krotoszynie": "Krotoszyn",
    "Łańcucie": "Łańcut",
    "Legionowie": "Legionowo",
    "Lesku": "Lesko",
    "Lesznie": "Leszno",
    "Limanowej": "Limanowa",
    "Lipnie": "Lipno",
    "Łobzie": "Łobez",
    "Łowiczu": "Łowicz",
    "Lubartowie": "Lubartów",
    "Lublinie": "Lublin",
    "Łukowie": "Łuków",
    "Mielcu": "Mielec",
    "Mińsku Mazowieckim": "Mińsk Mazowiecki",
    "Mławie": "Mława",
    "Mogilnie": "Mogilno",
    "Myślenicach": "Myślenice",
    "Myśliborzu": "Myślibórz",
    "Nisku": "Nisko",
    "Nowym Mieście Lubawskim": "Nowe Miasto Lubawskie",
    "Nowym Targu": "Nowy Targ",
    "Olecku": "Olecko",
    "Oleśnicy": "Oleśnica",
    "Oleśnie": "Olesno",
    "Olkuszu": "Olkusz",
    "Opatowie": "Opatów",
    "Opocznie": "Opoczno",
    "Opolu": "Opole",
    "Opolu Lubelskim": "Opole Lubelskie",
    "Ostrowcu Świętokrzyskim": "Ostrowiec Świętokrzyski",
    "Oświęcimiu": "Oświęcim",
    "Pabianicach": "Pabianice",
    "Pile": "Piła",
    "Pińczowie": "Pińczów",
    "Piotrkowie Trybunalskim": "Piotrków Trybunalski",
    "Pleszewie": "Pleszew",
    "Prudniku": "Prudnik",
    "Pruszkowie": "Pruszków",
    "Przasnyszu": "Przasnysz",
    "Przysusze": "Przysucha",
    "Puławach": "Puławy",
    "Raciborzu": "Racibórz",
    "Radomiu": "Radom",
    "Radziejowie": "Radziejów",
    "Rykach": "Ryki",
    "Rypinie": "Rypin",
    "Rzeszowie": "Rzeszów",
    "Siedlcach": "Siedlce",
    "Sławnie": "Sławno",
    "Słubicach": "Słubice",
    "Sochaczewie": "Sochaczew",
    "Sokółce": "Sokółka",
    "Śremie": "Śrem",
    "Starachowicach": "Starachowice",
    "Starogardzie Gdańskim": "Starogard Gdański",
    "Staszowie": "Staszów",
    "Strzelcach Opolskich": "Strzelce Opolskie",
    "Strzyżowie": "Strzyżów",
    "Świdnicy": "Świdnica",
    "Świdniku": "Świdnik",
    "Świebodzinie": "Świebodzin",
    "Szamotułach": "Szamotuły",
    "Szczytnie": "Szczytno",
    "Szubinie": "Szubin",
    "Tarnobrzegu": "Tarnobrzeg",
    "Tarnowie": "Tarnów",
    "Tomaszowie Mazowieckim": "Tomaszów Mazowiecki",
    "Turku": "Turek",
    "Wadowicach": "Wadowice",
    "Włoszczowie": "Włoszczowa",
    "Wieluniu": "Wieluń",
    "Wodzisławiu Śląskim": "Wodzisław Śląski",
    "Wołominie": "Wołomin",
    "Żaganiu": "Żagań",
    "Zamościu": "Zamość",
    "Żarach": "Żary",
    "Zielonej Górze": "Zielona Góra",
    "Złotowie": "Złotów",
    "Żyrardowie": "Żyrardów",
    "Żywcu": "Żywiec",
}

# -------- fallback suffix rules (very rough) --------
SUFFIX_RULES = [
    (r"aju\b",  "aj"),     # Biłgoraju -> Biłgoraj
    (r"owie\b", "ów"),     # Krakowie  -> Kraków
    (r"wskim\b","wskie"),  # Odrzańskim -> Odrzańskie
    (r"jcu\b",  "jec"),    # Grójcu    -> Grójec
    (r"pcu\b",  "piec"),   # Biskupcu  -> Biskupiec
    (r"wach\b", "wy"),     # Puławach  -> Puławy
    (r"nie\b",  "no"),     # Kępnie    -> Kępno
    (r"ciu\b",  "ć"),      # Mińsku    -> Mińsk
    (r"szu\b",  "sz"),     # Przasnyszu-> Przasnysz
    (r"lu\b",   "l"),      # Lublu?? – generic “lu” -> “l” (rare)
]

_preposition = re.compile(r"^\s*(w|we|na)\s+", re.I)

def locative_to_nominative(text: str | None) -> str | None:
    """
    Convert Polish locative/other oblique forms to nominative.

    1. Strip leading 'w', 'we', 'na' etc.
    2. Exact-match dictionary (LOC_TO_NOM)
    3. Regex suffix rules (SUFFIX_RULES)
    4. Fallback: return unchanged
    """
    if not isinstance(text, str):
        return None

    stripped = _preposition.sub("", text.strip())

    # dictionary first
    if stripped in LOC_TO_NOM:
        return LOC_TO_NOM[stripped]

    # heuristic suffixes
    for pat, repl in SUFFIX_RULES:
        if re.search(pat, stripped, flags=re.I):
            return re.sub(pat, repl, stripped, flags=re.I)

    return stripped


# ---- drop-in replacement for your earlier helper ----
def extract_city_from_rejonowa(rejonowa_cell: str | None) -> str | None:
    """
    Pull the toponym from strings like

        'Prokuratura Rejonowa w Radomiu'
        'Prokuratura Rejonowa Radom'

    …and return it in the **nominative**.
    """
    if not isinstance(rejonowa_cell, str) or not rejonowa_cell.strip():
        return None

    text = " ".join(rejonowa_cell.split())  # normalise spaces

    # 1️⃣   everything after the last ' w ' / ' we ' / ' na '
    m = re.search(r"\b(w|we|na)\s+([A-ZĄĆĘŁŃÓŚŹŻ].*)$", text)
    if m:
        loc = m.group(2)
    else:
        # 2️⃣ fallback: strip leading 'Prokuratura Rejonowa'
        loc = re.sub(r"(?i)^prokuratura rejonowa", "", text).strip(" ,.;")

    return locative_to_nominative(loc) if loc else None



























# ---------------------------------------------------------------------
# 1. Load the workbook
# ---------------------------------------------------------------------
SOURCE = "prokuratury_mapping-handfixed.xlsx"   # adjust path if necessary
df = pd.read_excel(SOURCE)

# ---------------------------------------------------------------------
# 2. Check that every gmina name is valid
# ---------------------------------------------------------------------
#   – must start with “gm.” or “m.” (case-insensitive)
#   – OR be one of the 18 Warsaw districts
#   – OR simply contain the word “Warszaw” (to catch variants like
#     “m. Warszawy”, “Dzielnica Wilanów m.st. Warszawy” etc.)

warsaw_districts = {
    "Bemowo", "Białołęka", "Bielany", "Mokotów", "Ochota",
    "Praga-Południe", "Praga-Północ", "Rembertów", "Śródmieście",
    "Targówek", "Ursus", "Ursynów", "Wawer", "Wesoła",
    "Wilanów", "Włochy", "Wola", "Żoliborz"
}
prefix_pat = re.compile(r"^(gm\.|m\.)\s", re.I)

def gmina_is_ok(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if prefix_pat.match(s):
        return True
    # strip possible prefixes and compare to the districts list
    stripped = s.replace("gm. ", "").replace("m. ", "").strip()
    if stripped in warsaw_districts:
        return True
    return bool(re.search(r"Warszaw", s, re.I))



def extract_city_from_rejonowaOLD(text: str | None) -> str | None:
    """
    Very simple heuristic:
        – look for the last ' w '  → take what follows
        – otherwise strip 'Prokuratura Rejonowa' and take the remainder
        – tidy whitespace and return None if nothing reasonable found
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # normalise multi-spaces
    t = " ".join(text.split())

    if " w " in t:
        city = t.split(" w ")[-1].strip(" ,.;")
    elif " we " in t:
        city = t.split(" w ")[-1].strip(" ,.;")
    else:
        city = re.sub(r"(?i)^prokuratura rejonowa", "", t).strip(" ,.;")

    return city if city else None


df["gmina_valid"] = df["gmina"].apply(gmina_is_ok)
invalid_gmina_rows = df.loc[~df["gmina_valid"]]

# ---------------------------------------------------------------------
# 3. Build the mapping according to your three rules
# ---------------------------------------------------------------------

def evaluate_one_gmina(group: pd.DataFrame) -> pd.Series:
    gmina = group.name
    rows_empty_opis    = group[group["opis"].isna()]
    rows_nonempty_opis = group[~group["opis"].isna()]

    anomalies = []
    chosen_okregowa = None

    # -- Rule 1
    if not rows_empty_opis.empty:
        ok_set = set(rows_empty_opis["prokuratura okręgowa"])
        if len(ok_set) == 1:
            chosen_okregowa = next(iter(ok_set))
        else:
            anomalies.append(
                f"Rule 1 violated – empty-opis rows have {sorted(ok_set)=}"
            )
        if not rows_nonempty_opis.empty:
            anomalies.append("Mix of empty and non-empty 'opis' rows")

    # -- Rule 2
    if rows_empty_opis.empty:
        opis_set  = set(rows_nonempty_opis["opis"])
        ok_set    = set(rows_nonempty_opis["prokuratura okręgowa"])
        if len(opis_set) >= 2 and len(ok_set) == 1:
            cand = next(iter(ok_set))
            if chosen_okregowa and chosen_okregowa != cand:
                anomalies.append("Rule 1 vs Rule 2 mismatch")
            chosen_okregowa = chosen_okregowa or cand
        else:
            anomalies.append("Rule 2 conditions unmet")

    return pd.Series(
        {"gmina": gmina,
         "chosen_okregowa": chosen_okregowa,
         "anomaly": "; ".join(anomalies) if anomalies else None}
    )

eval_df = (
    df.groupby("gmina", dropna=False)
      .apply(evaluate_one_gmina)
      .reset_index(drop=True)
)

# Split into clean vs anomalous gminas
good_gminas  = eval_df[eval_df["anomaly"].isna()].copy()
anomaly   = eval_df[eval_df["anomaly"].notna()].copy()

mapping_rows = []

for gmina, sub in df.groupby("gmina", dropna=False):

    # one or more distinct okręgowa values found for this gmina
    unique_ok = sorted(set(sub["prokuratura okręgowa"].dropna()))

    if not unique_ok:          # (extremely unlikely) no okręgowa at all
        continue

    # -----------------------------------------------------------
    # Case A: exactly one okręgowa  → leave the two new columns blank
    # -----------------------------------------------------------
    if len(unique_ok) == 1:
        mapping_rows.append(
            {
                "gmina": gmina,
                "prokuratura okręgowa": unique_ok[0],
                "prokuratura rejonowa": None,
                "siedziba rejonowa":   None,
            }
        )
    # -----------------------------------------------------------
    # Case B: several okręgowa  → replicate rows
    # -----------------------------------------------------------
    else:
        for ok in unique_ok:
            sub_ok = sub[sub["prokuratura okręgowa"] == ok]

            # pick (at most) one non-blank rejonowa for this pair
            rejonowa_vals = [
                x for x in sub_ok["prokuratura rejonowa"].dropna().unique()
                if str(x).strip()
            ]
            rejonowa = rejonowa_vals[0] if rejonowa_vals else None
            siedziba = extract_city_from_rejonowa(rejonowa)

            mapping_rows.append(
                {
                    "gmina": gmina,
                    "prokuratura okręgowa": ok,
                    "prokuratura rejonowa": rejonowa,
                    "siedziba rejonowa":   siedziba,
                }
            )


mapping_df = pd.DataFrame(mapping_rows)

# ---------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------
with pd.ExcelWriter("gmina_okregowa.xlsx") as writer:
    mapping_df.to_excel(writer, sheet_name="mapping",   index=False)
    anomaly.to_excel(writer, sheet_name="anomalies", index=False)

if not invalid_gmina_rows.empty:
    invalid_gmina_rows.to_excel("invalid_gmina_rows.xlsx", index=False)

print(
    f"✓ Mapping rows written:   {len(mapping_df):>4}\n"
    f"⚠️  Anomalous gminas:      {len(anomaly):>4}\n"
    f"⚠️  Invalid gmina names:   {len(invalid_gmina_rows):>4}"
)
