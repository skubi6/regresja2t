"""
commission_merge.py
───────────────────
Joins two PKW spreadsheets:

  • sklad_komisji_obwodowych_w_drugiej_turze_utf8.xlsx
  • OKW_2025_full.xlsx.gz2          (gzip-compressed .xlsx)

Creates one row per commission and, for every member, six attributes:
  name · candidate · uzupel · found · contradict

The order of members is           ◀ chairman,  deputy,  all others ▶
"""

from __future__ import annotations
import re, unicodedata
from pathlib import Path
import pandas as pd

# ──────────────── file paths ────────────────
#HERE       = Path(__file__).parent
#FILE_SKLAD = "sklad_komisji_obwodowych_w_drugiej_turze_utf8-mini.xlsx"
#FILE_OKW   = "OKW_2025_full-mini.xlsx"
#OUT_FILE   = "commission_combined-mini.xlsx"

FILE_SKLAD = "sklad_komisji_obwodowych_w_drugiej_turze_utf8.xlsx"
FILE_OKW   = "OKW_2025_full.xlsx"
OUT_FILE   = "commission_combined.xlsx"

# ─────────────── helper functions ───────────
ROLE_ORDER = {
    "Przewodniczący"                : 0,
    "Zastępca Przewodniczącego"     : 1,
}
CAND_RE    = re.compile(r"POLSKIEJ\s+([A-ZŁŚŻŹĆĄĘÓŃ ]+)", re.I)
STRIP_TOK  = r"\b(gm\.?|gmina|m\.)\b"

def strip_accents(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt)
                   if not unicodedata.combining(c))

count = 0
def norm_nameRev(txt: str) -> str:
    head, sep, tail = txt.rpartition(" ")
    return norm_name(f"{tail} {head}" if sep else txt)

def norm_name(txt: str) -> str:
    res= re.sub(r"\s+", " ",
                strip_accents(str(txt)).upper()).strip()
    #print ('<' + str(txt) + '> to <'+ res + '>')    
    return res

def norm_gmina(txt: str) -> str:
    txt = str(txt).split("\n", 1)[0]               # drop “Załącznik …”
    txt = re.sub(STRIP_TOK, "", txt, flags=re.I)
    res = re.sub(r"\s+", " ", strip_accents(txt).lower()).strip()
    #print ('<' + txt + '> to <'+ res + '>')
    return res

def candidate_nom(komitet: str | None) -> str | None:
    if not komitet:
        return None
    komitet = re.sub(r"\s*\(uzupełnienie.*", "", komitet, flags=re.I)
    m = CAND_RE.search(komitet)
    return m.group(1).title().strip() if m else None
# ────────────────────────────────────────────


# 1.  ──  Load both sources  ─────────────────
df_sklad = pd.read_excel(FILE_SKLAD)


#with gzip.open(FILE_OKW, "rb") as gz:
#    df_okw = pd.read_excel(gz.read())

df_okw = pd.read_excel(FILE_OKW)
print ('mark 2')
    
# # DEBUG – see a slice the user pointed at ---------------------------
# print(df_sklad.iloc[5677:5678, 0:1])
# --------------------------------------------------------------------

# 2.  ──  Normalise keys  ───────────────────
df_sklad["gmina_norm"] = df_sklad["Nazwa gminy"].map(norm_gmina)
df_sklad["name_norm"]  = df_sklad["Nazwisko i imiona"].map(norm_name)
df_sklad["comm_id"]    = list(zip(df_sklad["gmina_norm"],
                                  df_sklad["Nr obw."].astype(int)))

#print (list(zip(df_sklad["gmina_norm"],
#                                  df_sklad["Nr obw."].astype(int)))[:300])

df_okw["gmina_norm"]   = df_okw["gmina"].map(norm_gmina)
df_okw["name_norm"]    = df_okw["czlonek"].map(norm_nameRev)
df_okw["is_uzup"]      = df_okw["komitet"].str.contains("uzupełnienie",
                                                        case=False, na=False)
df_okw["candidate"]    = df_okw["komitet"].map(candidate_nom)
df_okw["comm_id"]      = list(zip(df_okw["gmina_norm"],
                                  df_okw["komisja_nr"].astype(int)))

#print (list(zip(df_okw["gmina_norm"],
#                df_okw["komisja_nr"].astype(int)))[:300])

# group OKW rows commission-wise for O(1) lookup
okw_by_comm = {cid: grp for cid, grp in df_okw.groupby("comm_id")}

#print ()
#print ('groups')
#print (okw_by_comm)

# 3.  ──  Build output rows  ────────────────
out_rows  : list[dict] = []












# ---------------- ➊ helper: fast index by (powiat, nr_obwodu) -----------
# build once, right after df_okw is ready
okw_powiat_idx: dict[tuple[str, int], pd.DataFrame] = {}

for (powiat, nr), grp in df_okw.groupby(["gmina_norm", "komisja_nr"]):
    # we’ll also reuse these groups grouped by POWIAT later, so save them …
    key = (powiat, nr)
    okw_powiat_idx[key] = grp

#print ()
#print ('okw_powiat_idx')
#print (okw_powiat_idx)

# ------------- ➋ new helper: smart lookup ------------------------------
def find_okw_group(
    gmina_norm: str,
    powiat_norm: str,
    nr_obw: int
) -> pd.DataFrame | None:
    """
    1. exact gmina + nr_obw  (old behaviour)
    2. any row in the same POWIAT with the same nr_obw
    """
    exact = okw_by_comm.get((gmina_norm, nr_obw))
    if exact is not None and not exact.empty:
        return exact

    fallback = okw_powiat_idx.get((powiat_norm, nr_obw))
    return fallback


# --------------------- ➌ the corrected loop -----------------------------
rows = []

for cid, part in df_sklad.groupby("comm_id"):
    base = part.iloc[0][["TERYT gminy", "Nazwa gminy",
                         "Powiat", "Nr obw."]].to_dict()
    #print ('base', base)
    powiat_norm = strip_accents(str(base["Powiat"]).lower())
    nr_obw      = int(base["Nr obw."])

    part_sorted = part.sort_values(
        by=["Funkcja", "Nazwisko i imiona"],
        key=lambda s: s.map(lambda x: ROLE_ORDER.get(x, 2))
    )

    # NEW ─ get OKW rows once per commission
    okw_here = find_okw_group(
        gmina_norm=part.iloc[0]["gmina_norm"],
        powiat_norm=powiat_norm,
        nr_obw=nr_obw,
    )

    for idx, mem in enumerate(part_sorted.to_dict("records"), start=1):
        name_norm = mem["name_norm"]

        # ---------- matching logic ----------
        found, uzup, contradict = False, False, False
        cand = None

        if okw_here is not None:
            match = okw_here[okw_here["name_norm"] == name_norm]

            if match.empty:
                # surname prefix (still uppercase/stripped) as fuzzy fallback
                surname = name_norm.split()[0]
                match = okw_here[okw_here["name_norm"].str.startswith(surname)]

            if not match.empty:
                found = True
                uzup  = bool(match["is_uzup"].any())

                cand_list = match["candidate"].dropna().unique().tolist()
                if len(cand_list) == 1:
                    cand = cand_list[0]
                elif len(cand_list) > 1:
                    cand = ", ".join(cand_list)
                    contradict = True
        # ---------- write columns -----------
        pre = f"member{idx}"
        base.update({
            f"{pre}_name"      : mem["Nazwisko i imiona"],
            f"{pre}_candidate" : cand,
            f"{pre}_uzup"      : uzup,
            f"{pre}_found"     : found,
            f"{pre}_contradict": contradict,
        })
    rows.append(base)

df_out = pd.DataFrame(rows)
df_out.to_excel(OUT_FILE, index=False, engine="xlsxwriter")
print(f"✓  {len(df_out):,} commissions written to {OUT_FILE}")
