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

df["gmina_valid"] = df["gmina"].apply(gmina_is_ok)
invalid_gmina_rows = df.loc[~df["gmina_valid"]]

# ---------------------------------------------------------------------
# 3. Build the mapping according to your three rules
# ---------------------------------------------------------------------
def evaluate_one_gmina(group: pd.DataFrame) -> pd.Series:
    """
    Implements the logic you specified.

    Returns
    -------
    pd.Series with:
        • 'gmina'                 – the group key
        • 'prokuratura okręgowa'  – the resolved value or None
        • 'anomaly'               – textual description or None
    """
    gmina = group.name
    rows_empty_opis     = group[group["opis"].isna()]
    rows_nonempty_opis  = group[~group["opis"].isna()]

    anomalies = []
    chosen_okregowa = None

    # ---------- Rule 1: at least one row with opis empty ----------
    if not rows_empty_opis.empty:
        okreg_set = set(rows_empty_opis["prokuratura okręgowa"])
        if len(okreg_set) == 1:
            chosen_okregowa = okreg_set.pop()
        else:
            anomalies.append(
                f"Rule 1 violated – different okręgowa values in empty-opis rows: {sorted(okreg_set)}"
            )
        if not rows_nonempty_opis.empty:
            anomalies.append("Empty and non-empty 'opis' rows mixed")

    # ---------- Rule 2: NO row with opis empty ----------
    if rows_empty_opis.empty:
        opis_set   = set(rows_nonempty_opis["opis"])
        okreg_set  = set(rows_nonempty_opis["prokuratura okręgowa"])
        if len(opis_set) >= 2 and len(okreg_set) == 1:
            candidate = next(iter(okreg_set))
            if chosen_okregowa and chosen_okregowa != candidate:
                anomalies.append(
                    "Conflict – Rule 1 and Rule 2 yield different okręgowa values"
                )
            chosen_okregowa = chosen_okregowa or candidate
        else:
            anomalies.append(
                "Rule 2 conditions not met (need ≥2 different non-empty 'opis' values and exactly one okręgowa)"
            )

    return pd.Series({
        "gmina": gmina,
        "prokuratura okręgowa": chosen_okregowa,
        "anomaly": "; ".join(anomalies) if anomalies else None
    })

result = (
    df.groupby("gmina", dropna=False)
      .apply(evaluate_one_gmina)
      .reset_index(drop=True)
)

mapping_df   = result[result["anomaly"].isna()].drop(columns="anomaly")
anomaly_df   = result[result["anomaly"].notna()]

# ---------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------
with pd.ExcelWriter("gmina_okregowa.xlsx") as writer:
    mapping_df.to_excel(writer, sheet_name="mapping",   index=False)
    anomaly_df.to_excel(writer, sheet_name="anomalies", index=False)

if not invalid_gmina_rows.empty:
    invalid_gmina_rows.to_excel("invalid_gmina_rows.xlsx", index=False)

print(
    f"✓ Mapping rows written:   {len(mapping_df):>4}\n"
    f"⚠️  Anomalous gminas:      {len(anomaly_df):>4}\n"
    f"⚠️  Invalid gmina names:   {len(invalid_gmina_rows):>4}"
)
