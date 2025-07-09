#!/usr/bin/env python3
# ------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import mahalanobis
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import statistics
from datetime import datetime
import argparse
from math import sqrt
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import re

nowStart = datetime.now()

inputFileNames = {
    2025: ("protokoly_po_obwodach_utf8-fixed.xlsx",
           "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx",
           "commission_combined.xlsx",
           "obwody_geo.xlsx"),
    2020: ("2020-t1-wyniki_gl_na_kand_po_obwodach_utf8.xlsx",
           "2020-t2-wyniki_gl_na_kand_po_obwodach_utf8.xlsx",
           None),
    2015: ("2015-wyniki_tura1-1.xls", "2015-wyniki_tura2.xls", None)
}

outputFileNames = {
    2025: ("protokoly_2t_wzbogacone_2025.xlsx",)
}

prokFileName = 'prok/gmina_okregowa.xlsx'

terytGminy = {
    2025: "Teryt Gminy",
    2020: "Kod TERYT"
}

nrKomisji = {
    2025: "Nr komisji",
    2020: "Numer obwodu"
}

def ci_diffL(N, p, q, x=0.95):
    """Przedział ufności dla A-B przy pewności x"""
    mu   = diff_mean(N, p, q)
    sig  = sqrt(diff_var(N, p, q))
    z    = norm.ppf((1 + x) / 2)       # kwantyl 1-α/2
    return mu - z*sig

def ci_diffH(N, p, q, x=0.95):
    """Przedział ufności dla A-B przy pewności x"""
    mu   = diff_mean(N, p, q)
    sig  = sqrt(diff_var(N, p, q))
    z    = norm.ppf((1 + x) / 2)       # kwantyl 1-α/2
    return mu + z*sig

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius [km]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))



# ------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------
def nan_diagnostic(X, y, tgt_name, KEY1, KEY2):
    ignore = ['Powiat', 'Teryt Powiatu', 'Województwo', 'Gmina', 'Nr OKW', 'Typ gminy']
    cols_with_nan = (
        X.columns[X.isna().any()]          # columns that have at least one NaN
        .difference(ignore)              # remove the ones we want to skip
        .tolist()
    )
    y_nan = y.isna().any()
    if not cols_with_nan and not y_nan:
        return
    mask = X[cols_with_nan].isna().any(axis=1) | y.isna()
    offenders = (
        pd.concat([X.loc[mask, cols_with_nan], y.loc[mask]], axis=1)
        .reset_index()[[KEY1, KEY2] + cols_with_nan + [tgt_name]]
    )
    offendersS =  pd.ExcelWriter('offenders.xlsx', engine="xlsxwriter")
    offenders.to_excel(offendersS, sheet_name='offenders', index=True)
    print ("# offenders", offenders.shape[0])
    X.to_excel(offendersS, sheet_name='X', index=True)
    y.to_excel(offendersS, sheet_name='y', index=True)
    X.loc[mask, cols_with_nan].to_excel(offendersS, sheet_name='X.loc(mask, cols_with_nan)', index=True)
    y.loc[mask].to_excel(offendersS, sheet_name='y.loc(mask)', index=True)
    X[cols_with_nan].isna().any(axis=1).to_excel(offendersS, sheet_name="maskX", index=True)
    y.isna() .to_excel(offendersS, sheet_name="masky", index=True)
    mask.to_excel(offendersS, sheet_name="mask", index=True)
    offendersS.close()
    print ('cols_with_nan', cols_with_nan)
    print('cols X')
    print( X.columns.tolist())
    print ('tgt_name', tgt_name)
    raise "nans"
    keys = offenders[[KEY1, KEY2]].astype(str).agg(" · ".join, axis=1)
    print(f"\n❌ NaNs in target '{tgt_name}':")
    for k in keys:
        print("  -", k)
    offenders.to_excel(f"nan_offenders_{tgt_name}.xlsx", index=False)
    sys.exit(1)

def assert_no_dupes(df, key_cols, label):
    dup = df.duplicated(key_cols, keep=False)
    if dup.any():
        raise SystemExit(
            f"\n❌ DUPLICATE KEYS in {label}:\n"
            f"{df.loc[dup, key_cols].to_string(index=False)}\n"
        )

# ------------------------------------------------------------
# 2.  Load Excel sheets
# ------------------------------------------------------------
DATA_DIR = Path(".")

cands = {
    2025: [
        "NAWROCKI Karol Tadeusz",  # c1
        "TRZASKOWSKI Rafał Kazimierz",  # c2
        "BARTOSZEWICZ Artur",
        "BIEJAT Magdalena Agnieszka",
        "BRAUN Grzegorz Michał",
        "HOŁOWNIA Szymon Franciszek",
        "JAKUBIAK Marek",
        "MACIAK Maciej",
        "MENTZEN Sławomir Jerzy",
        "SENYSZYN Joanna",
        "STANOWSKI Krzysztof Jakub",
        "WOCH Marek Marian",
        "ZANDBERG Adrian Tadeusz",
    ],
    2020 : [
        "Robert BIEDROŃ",
        "Krzysztof BOSAK",
        "Andrzej Sebastian DUDA",
	"Szymon Franciszek HOŁOWNIA",
	"Marek JAKUBIAK",
        "Władysław Marcin KOSINIAK-KAMYSZ",
	"Mirosław Mariusz PIOTROWSKI",
	"Paweł Jan TANAJNO",
	"Rafał Kazimierz TRZASKOWSKI",
	"Waldemar Włodzimierz WITKOWSKI",
	"Stanisław Józef ŻÓŁTEK"
    ]
}

first_cols = {
    2025: [
        'Liczba wyborców uprawnionych do głosowania (umieszczonych w\xa0spisie, z\xa0uwzględnieniem dodatkowych formularzy) w\xa0chwili zakończenia głosowania',
        'w tym z powodu postawienia znaku „X” obok nazwiska dwóch lub większej liczby kandydatów',
        'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania',
        'Liczba głosów nieważnych (z\xa0kart ważnych)'
        ] + cands[2025],
    2020: [
        'Liczba wyborców uprawnionych do głosowania',
        'Liczba wyborców głosujących na podstawie zaświadczenia o prawie do głosowania',
        'Liczba głosów nieważnych'
    ] + cands[2020]
}

c = {
        2025 : ["NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"],
        2020 : ["Andrzej Sebastian DUDA", "Rafał Kazimierz TRZASKOWSKI"]
}

targets = {
    2025: [
        'Liczba głosów nieważnych (z\xa0kart ważnych)',
        'w tym z powodu postawienia znaku „X” obok nazwisk obu kandydatów',
        "NAWROCKI Karol Tadeusz",          # c1
        "TRZASKOWSKI Rafał Kazimierz",     # c2
    ],
        
    2020: [
        'Liczba głosów nieważnych',
        #'w tym z powodu postawienia znaku „X” obok nazwisk obu kandydatów'
        "Andrzej Sebastian DUDA",          # c1
        "Rafał Kazimierz TRZASKOWSKI",     # c2
    ]
}

targetTranslate = {
        'Liczba głosów nieważnych (z\xa0kart ważnych)' : 'nieważne',
        'w tym z powodu postawienia znaku „X” obok nazwisk obu kandydatów' : 'dwa X',
        "NAWROCKI Karol Tadeusz" : 'NAWROCKI',          # c1
        "TRZASKOWSKI Rafał Kazimierz" : 'TRZASKOWSKI',     # c2
        "Andrzej Sebastian DUDA" : 'DUDA',          # c1
        "Rafał Kazimierz TRZASKOWSKI" : 'TRZASKOWSKI',     # c2
}

# = "Liczba wyborców głosujących na podstawie zaświadczenia o prawie do głosowania" # z tabelki drugiej tury
# = 'Liczba wyborców głosujących na podstawie zaświadczenia o prawie do głosowania'
EXTRApredictor = 'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania'
# ------------------------------------------------------------
# 3.  Design & target matrices
# ------------------------------------------------------------


#Y = SECOND.set_index([KEY1, KEY2])

def buildAggregate (
        Xarg, Yarg, filename, rok
):
    nowFunStart = datetime.now()

    #print ('Xarg.index')
    #print (Xarg.index)
    #print ('Yarg.index')
    #print (Yarg.index)
    
    Xarg, Yarg = Xarg.align(Yarg, join="inner", axis=0)
    print("Precincts analysed:", len(Xarg))

    Yarg.reset_index(inplace=True)

    writerY = pd.ExcelWriter(filename, engine="xlsxwriter")
    Yarg.to_excel(writerY, sheet_name="Y", index=False)

    writerY.close()
    nowFunStop = datetime.now()

def main():
    parser = argparse.ArgumentParser(
        description="Demo: accept a '-c' flag plus positional arguments"
    )

    parser.add_argument(
        'items',
        nargs='*',
        help='List of positional arguments'
    )

    args = parser.parse_args()
    rok = 2025
    if 0<len(args.items):
        rok = int (args.items[0])
    print ('ROK', rok)
    global KEY1
    global KEY2
    KEY1, KEY2 = terytGminy[rok], nrKomisji[rok]

    FIRST = pd.read_excel(DATA_DIR / inputFileNames[rok][0],
                          dtype={'Teryt Gminy': "Int64"})
    FIRST.loc[FIRST['Typ obszaru'].isin(['zagranica', 'statek']), terytGminy[rok]] = 9999999
    SECOND = pd.read_excel(DATA_DIR / inputFileNames[rok][1],
                          dtype={'Teryt Gminy': "Int64"})
    print ('SECOND key col')
    print (SECOND[EXTRApredictor])
    print ('SECOND cols')
    print (list(SECOND.columns))
    SECOND = SECOND.rename (columns={EXTRApredictor : 'zaswiadczenia2t'})
    print (list(SECOND.columns))
    SECOND.loc[SECOND['Typ obszaru'].isin(['zagranica', 'statek']), terytGminy[rok]] = 9999999
    GUS = pd.read_excel(
            "powierzchnia_i_ludnosc_w_przekroju_terytorialnym_w_2024_roku_tablice.xlsx",
            sheet_name="Tabl. 21",
            skiprows=3,        # drop the 5 rows *above* the real header
            usecols="A:F")
    GUS.columns = ['Teryt Gminy', 'nazwa gminy (GUS)', 'powierzchnia', 'D', 'E', 'Ludnosc']
    GUS = GUS.drop(columns=["D", "E"])
    #GUS['Teryt Gminy']  = GUS['Teryt Gminy'] + '01'
    #GUS = GUS[GUS['Teryt Gminy'].str.rsplit(n=1, expand=True).iloc[:, 1].isin(["1", "2", "3"])]
    pattern = r"\s[123]$"
    GUS = GUS[GUS["Teryt Gminy"].str.contains(pattern, na=False)]
    GUS["Teryt Gminy"] = GUS["Teryt Gminy"].str.replace(pattern, "", regex=True)
    GUS["Teryt Gminy"] = GUS["Teryt Gminy"].str.replace(r"^0", "", regex=True)
    FIRST['Teryt Gminy']  = FIRST['Teryt Gminy'].astype(str)
    SECOND['Teryt Gminy']  = SECOND['Teryt Gminy'].astype(str)
    #print ("SECOND['Teryt Gminy']")
    #print (SECOND['Teryt Gminy'])
    #print ('GUS')
    #print(GUS['Teryt Gminy'])
    SECOND = SECOND.merge(
        GUS,
        on='Teryt Gminy',
        how='left'
    )
    SECOND.loc[SECOND['Powiat']=='Warszawa', 'Ludnosc'] = 1863056

    GEO = pd.DataFrame()
    if inputFileNames[rok][3]:
        GEO = pd.read_excel(DATA_DIR / inputFileNames[rok][3])
        GEO.rename(columns={"Numer": "Nr komisji", "TERYT gminy" : "Teryt Gminy"}, inplace=True)
        GEO[KEY1] = GEO[KEY1].astype(str)
    if 0 < GEO.shape[0]:
        SECONDmerged = SECOND.merge(GEO, on=[KEY1, KEY2], how="left",
                                     indicator=True,
                                     suffixes=("", "_extra"),
                                     validate="many_to_one")
        mask_missing = (SECONDmerged['_merge'] == 'left_only') & (SECONDmerged['Teryt Gminy'] != '9999999')
        if mask_missing.any():
            bad_keys = (
                SECONDmerged.loc[mask_missing, [KEY1, KEY2]]
                .drop_duplicates()
                .to_dict('records')
            )
            raise KeyError(
                f"{mask_missing.sum()} row(s) in SECOND had no match in GEO for "
                f"({KEY1!s}, {KEY2!s}) pairs: {bad_keys}"
            )
        SECOND = SECONDmerged.drop(columns=['_merge'])

        BLACKLIST = { # Teryt gmin, w kórych na pewno nie ma prokuratur rejonowych
            "120405",
        }

        SEAT_COORDS = {}
        mapping_df = pd.read_excel(prokFileName, sheet_name="mapping")
        for seat_city in mapping_df["siedziba rejonowa"].dropna().unique():
            mask = SECOND["Gmina"].str.fullmatch(
                rf"m\. {re.escape(seat_city)}", na=False
            ) & ~SECOND["Teryt Gminy"].astype(str).isin(BLACKLIST)
            if not mask.any():
                mask = SECOND["Gmina"].str.fullmatch(
                    rf"gm\. {re.escape(seat_city)}", na=False
                ) & ~SECOND["Teryt Gminy"].astype(str).isin(BLACKLIST)
            if not mask.any():
                raise ValueError(
                    f"Seat city '{seat_city}' not found as 'm. {seat_city}' or 'gm. {seat_city}' in SECOND."
                )
            rows = SECOND[mask]
            SEAT_COORDS[seat_city] = list(
                rows[["lat", "lng"]].itertuples(index=False, name=None)
            )
        # helper: distance between a (lat,lng) and the seat of a given okręgowa
        def min_dist_to_okregowa(lat, lng, okr_row):
            seat_city = okr_row["siedziba rejonowa"]
            coords = SEAT_COORDS[seat_city]
            return min(haversine(lat, lng, lat2, lng2) for lat2, lng2 in coords)

        g2rows = (
            mapping_df.groupby("gmina")
            .apply(lambda x: x.to_dict(orient="records"))
            .to_dict()
        )
        def choose_okregowa(row):
            if str(row["Teryt Gminy"]) == "999999":
                return None

            gmina = row["Gmina"]
            powiat = row["Powiat"]
            candidates = g2rows.get(gmina, [])

            # no entry at all → leave blank (or raise, if you prefer)
            if not candidates:
                return None

            # 4.2  exactly one candidate – trivial join
            if len(candidates) == 1:
                return candidates[0]["prokuratura okręgowa"]

            if gmina.strip() == "gm. Czarna" and powiat.strip() == 'łańcucki':
                for cand in candidates:
                    if cand["siedziba rejonowa"] == "Lesko":
                        return cand["prokuratura okręgowa"]
            
            # 4.3  several candidates – pick the geographically nearest seat
            lat, lng = row["lat"], row["lng"]
            dists = [
                (min_dist_to_okregowa(lat, lng, cand), cand["prokuratura okręgowa"], cand["siedziba rejonowa"])
                for cand in candidates
            ]
            dists.sort(key=lambda x: x[0])               # nearest first

            # enforce “nearest : next” ≥ 1 : 2
            if len(dists) >= 2 and dists[0][0] * 2 > dists[1][0]:
                raise ValueError (
                    f"Gmina '{gmina}': distance ratio {dists[0][0]:.1f} km vs "
                    f"{dists[1][0]:.1f} km < 1:2; {dists[0][1]} vs. {dists[1][1]}; {dists[0][2]} vs. {dists[1][2]}; {powiat}"
                )

            return dists[0][1]
        
        SECOND["prokuratura"] = SECOND.apply(choose_okregowa, axis=1)

    EXTRA = pd.DataFrame()
    if inputFileNames[rok][2]:
        EXTRA = pd.read_excel(DATA_DIR / inputFileNames[rok][2])
        EXTRA.rename(columns={"Nr obw.": "Nr komisji", "TERYT gminy" : "Teryt Gminy"}, inplace=True)
        EXTRA[KEY2] = EXTRA[KEY2].replace("", 0).astype(int)
    for df in (FIRST, SECOND):
        df[KEY2] = df[KEY2].replace("", 0).astype(int)
    FIRST[KEY1] = FIRST[KEY1].fillna(0).astype(int)
    if 0 < EXTRA.shape[0]:
        EXTRA[KEY1] = EXTRA[KEY1].astype(str)
        #print ('SECOND before[KEY1]')
        #print (SECOND[KEY1])
        #print ('SECOND before[KEY2]')
        #print (SECOND[KEY2])

        #print ('EXTRA[KEY1]')
        #print (EXTRA[KEY1])
        #print ('EXTRA[KEY2]')
        #print (EXTRA[KEY2])
        SECOND_JOIN = (
            SECOND.merge(EXTRA, on=[KEY1, KEY2], how="inner",
                         suffixes=("", "_extra"))
        )

        print("Rows in joined SECOND+EXTRA:", len (SECOND), len(SECOND_JOIN))
        SECOND = SECOND_JOIN
        #print ('SECOND after[KEY1]')
        #print (SECOND[KEY1])
        #print ('SECOND after[KEY2]')
        #print (SECOND[KEY2])
        SECOND[KEY1] = SECOND[KEY1].fillna(0).astype(int)    

    print ('step 1')

    assert_no_dupes(FIRST, [KEY1, KEY2], "FIRST")
    assert_no_dupes(SECOND, [KEY1, KEY2], "SECOND")

    X = FIRST.set_index([KEY1, KEY2])
    #print ('SECOND')
    #print (SECOND)
    Y = SECOND.set_index([KEY1, KEY2])
    #print("Columns in SECOND:", SECOND.columns.tolist())
    print("Columns in Y:", Y.columns.tolist())

    buildAggregate (X, Y, outputFileNames[rok][0], rok)
    
if __name__ == "__main__":
    main()
