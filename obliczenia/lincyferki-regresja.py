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
#from sklearn.linear_model import RidgeCV
import re
from sklearn.linear_model import (
    RidgeCV,
    LinearRegression,
    HuberRegressor,      #  <-- already needed for the robust branch
    RANSACRegressor,     #  <-- this line cures the NameError
)

nowStart = datetime.now()

inputFileNames = {
    2025: ("protokoly_po_obwodach_utf8-fixed.xlsx", "protokoly_2t_wzbogacone_2025.xlsx"),
}

DATA_DIR = Path(".")

terytGminy = {
    2025: "Teryt Gminy",
    2020: "Kod TERYT"
}

nrKomisji = {
    2025: "Nr komisji",
    2020: "Numer obwodu"
}

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

# ------------------------------------------------------------
# 3.  Design & target matrices
# ------------------------------------------------------------


#Y = SECOND.set_index([KEY1, KEY2])

def buildLinRegression (
        Xarg, Yarg, filename, rok, *, t2absentee=False, robust=False
):
    nowFunStart = datetime.now()

    Xarg, Yarg = Xarg.align(Yarg, join="inner", axis=0)
    print("Precincts analysed:", len(Xarg))

    intercepts = {}
    coefs, fits, resids = {}, {}, {}

    ALPHAS = np.logspace(-3, 3, 13)      # 1e-3 … 1e3
    
    for tgt in targets[rok]:
        nan_diagnostic(Xarg, Yarg[tgt], tgt, KEY1, KEY2)
        
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),    # 1. imputacja braków
            StandardScaler(),                    # 2. pełna standaryzacja
            (RANSACRegressor(
                estimator=HuberRegressor(epsilon=1.35, alpha=0.0),
                min_samples=0.5,  # tolerate up to 50 % arbitrary rows
                residual_threshold=None,  # auto: MAD‑based
                max_trials=100,
                random_state=0,
            ) if robust else RidgeCV(alphas=ALPHAS, cv=5))         # 3. regresja z CV po alfach
        )

        if t2absentee:
            Xfeat = Xarg[first_cols[rok]].join(Yarg[['zaswiadczenia2t']])
        else:
            Xfeat = Xarg[first_cols[rok]]
                
        try:
            pipe.fit(Xfeat, Yarg[tgt])
        except Exception:
            writerD = pd.ExcelWriter("debug.xlsx", engine="xlsxwriter")
            Xarg.to_excel(writerD, sheet_name="Xarg", index=True)
            Yarg[[tgt]].to_excel(writerD, sheet_name="Yarg", index=True)
            writerD.close()
            raise

        Yarg["fits" + tgt] = pd.Series(
            pipe.predict(Xfeat),
            index=Xfeat.index
        )
        Yarg["resids" + tgt] = Yarg[tgt] - Yarg["fits" + tgt]

        # -------- współczynniki w ORYGINALNEJ skali cech ------------------------
        scaler = pipe.named_steps["standardscaler"]

        if robust:
            final_est = pipe.named_steps.get("ransacregressor", pipe.named_steps.get("ridgecv"))
            est = final_est.estimator_ if hasattr(final_est, "estimator_") else final_est
            beta_std = est.coef_
            beta_orig = beta_std / scaler.scale_  # un‑standardise
            
            intercept_orig = est.intercept_ - np.sum(scaler.mean_ * beta_orig)       # korekta interceptu
        else:
            ridge  = pipe.named_steps["ridgecv"]
            beta_std = ridge.coef_                       # współczynniki po skalowaniu
            beta_orig = beta_std / scaler.scale_         # „od-standaryzowanie”

            intercept_orig = ridge.intercept_ - np.sum(scaler.mean_ * beta_orig)       # korekta interceptu
        

        coefs[targetTranslate[tgt]]      = beta_orig
        intercepts[targetTranslate[tgt]] = intercept_orig

    predictor_names = ["Intercept"] + first_cols[rok]
    if t2absentee:
        predictor_names += ['zaswiadczenia2t']
    coef_tbl = pd.DataFrame(index=predictor_names)

    for tgt in targets[rok]:
        coef_tbl[targetTranslate[tgt]] = [intercepts[targetTranslate[tgt]]] + coefs[targetTranslate[tgt]].tolist()

    # convert to percent & round to 4 decimal places
    coef_tblStdout = (coef_tbl * 100).round(2)

    # custom float formatter for aligned output
    float_fmt = lambda x: f"{x:10.2f}"

    print("\n================ Linear-Hypothesis Coefficients (% units) ================")
    print(coef_tblStdout.to_string(float_format=float_fmt))
    print("==========================================================================\n")


    c1, c2 = c[rok][0], c[rok][1]

    Yarg["obs_diff"] = Yarg[c1] - Yarg[c2]
    Yarg["fit_diff"] = Yarg['fits' + c1] - Yarg['fits' + c2]
    Yarg["D"] = Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex
    Yarg["Drev"] = -Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex

    Yarg["Dnaw"] = Yarg[c1] - Yarg['fits' + c1]
    Yarg["Dtrza"] = Yarg[c2] - Yarg['fits' + c2]

    N   = Xarg[first_cols[rok][0]]
    p   = Yarg["fits" + c1] / N
    q   = Yarg["fits" + c2] / N

    Yarg["p"] = p
    Yarg["q"] = q

    # --- parametry rozkładu różnicy Nawrocki-Trzaskowski, Nawrockiego, Trzaskowskiego
    mu   = N * (p - q)                                # E[A-B]
    muNaw = N * p
    muTrza = N * q
    var  = N * (p + q - (p - q)**2)                   # Var[A-B]
    sigma = np.sqrt(var)                              # odchylenie std.

    Yarg["diff_mu"]   = mu
    Yarg["diff_var"]  = var
    Yarg["diff_std"]  = sigma

    varNaw  = N * p * (1 - p)          # Var[A]
    varTrza = N * q * (1 - q)          # Var[B]
    stdNaw  = np.sqrt(varNaw)
    stdTrza = np.sqrt(varTrza)
    
    Yarg["naw_mu"]   = muNaw           # E[A]
    Yarg["naw_var"]  = varNaw
    Yarg["naw_std"]  = stdNaw
    
    Yarg["trza_mu"]  = muTrza          # E[B]
    Yarg["trza_var"] = varTrza
    Yarg["trza_std"] = stdTrza

    for conf, tag in [(0.95, "95"), (0.995, "995"), (0.9995, "9995")]:
        z = norm.ppf(0.5 + conf / 2)
        Yarg[f"naw_ci{tag}_low"]  = muNaw  - z * stdNaw
        Yarg[f"naw_ci{tag}_high"] = muNaw  + z * stdNaw
        Yarg[f"trza_ci{tag}_low"] = muTrza - z * stdTrza
        Yarg[f"trza_ci{tag}_high"] = muTrza + z * stdTrza
        Yarg[f"diff_ci{tag}_low"]  = mu - z * sigma
        Yarg[f"diff_ci{tag}_high"] = mu + z * sigma
    
    # --- 95 % przedział ufności ---------------------------------------------------

    #conf = 0.95
    #z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    #Yarg["diff_ci95_low"]  = mu - z * sigma
    #Yarg["diff_ci95_high"] = mu + z * sigma

    #conf = 0.995
    #z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    #Yarg["diff_ci995_low"]  = mu - z * sigma
    #Yarg["diff_ci995_high"] = mu + z * sigma

    #conf = 0.9995
    #z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    #Yarg["diff_ci9995_low"]  = mu - z * sigma
    #Yarg["diff_ci9995_high"] = mu + z * sigma

    # --- jaki poziom ufności miałby przedział z brzegiem w obserwowanym D --------
    z_edge           = (Yarg["obs_diff"] - mu).abs() / sigma
    Yarg["x_edge"]   = 2 * norm.cdf(z_edge) - 1       # x ∈ (0, 1)

    z_naw_edge  = (Yarg[c1] - muNaw).abs()  / stdNaw
    z_trza_edge = (Yarg[c2] - muTrza).abs() / stdTrza
    Yarg["naw_edge"]  = 2 * norm.cdf(z_naw_edge)  - 1   # ∈ (0, 1)
    Yarg["trza_edge"] = 2 * norm.cdf(z_trza_edge) - 1
    

    denom   = (Yarg[c1] + Yarg[c2])**.5

    D_rel = pd.Series(                                              # keep original index
        np.divide(                                                  # element-wise D / denom
            Yarg["D"].to_numpy(dtype=float),                        # numerator
            denom.to_numpy(dtype=float),                            # denominator
            out=np.zeros_like(denom, dtype=float),                  # preset output with 0s
            where=~((Yarg["D"] == 0) & (denom == 0))                # skip rows where both are 0
        ),
        index=Yarg.index
    )
    
    D_rel   = D_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    D_rel   = D_rel.dropna()
    Yarg["Dnorm"] = D_rel

    Dnaw_rel = pd.Series(                                              # keep original index
        np.divide(                                                  # element-wise D / denom
            Yarg["Dnaw"].to_numpy(dtype=float),                        # numerator
            denom.to_numpy(dtype=float),                            # denominator
            out=np.zeros_like(denom, dtype=float),                  # preset output with 0s
            where=~((Yarg["Dnaw"] == 0) & (denom == 0))                # skip rows where both are 0
        ),
        index=Yarg.index
    )
    
    Dnaw_rel   = Dnaw_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    Dnaw_rel   = Dnaw_rel.dropna()
    Yarg["Dnaw_norm"] = Dnaw_rel

    Dtrza_rel = pd.Series(                                              # keep original index
        np.divide(                                                  # element-wise D / denom
            Yarg["Dtrza"].to_numpy(dtype=float),                        # numerator
            denom.to_numpy(dtype=float),                            # denominator
            out=np.zeros_like(denom, dtype=float),                  # preset output with 0s
            where=~((Yarg["Dtrza"] == 0) & (denom == 0))                # skip rows where both are 0
        ),
        index=Yarg.index
    )
    
    Dtrza_rel   = Dtrza_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    Dtrza_rel   = Dtrza_rel.dropna()
    Yarg["Dtrza_norm"] = Dtrza_rel
  
    #Drev_rel   = Yarg["D"] / denom
    #Drev_rel   = Drev_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    #Drev_rel   = Drev_rel.dropna()
    #Yarg["Drevnorm"] = D_rel

    nowAfterInit = datetime.now()

    Yarg.reset_index(inplace=True)

    writerY = pd.ExcelWriter(filename, engine="xlsxwriter")
    Yarg.to_excel(writerY, sheet_name="Y", index=False)
    coef_tbl.to_excel(writerY, sheet_name="Coefficients", index=True)
    ws  = writerY.sheets["Coefficients"]   # the new worksheet
    wb  = writerY.book

    # 1) show numbers as 12.34 %
    percent_fmt = wb.add_format({"num_format": "0.00%"})
    
    # 2) set column widths
    #    Excel’s width unit ≈ width of one “0” → ~0.19 cm with Calibri 11
    #    → 5 cm ≈ 27 units, 2 cm ≈ 11 units
    ws.set_column(0, 0, 20)                                # first col (labels)  ≈ 5 cm
    ws.set_column(1, coef_tbl.shape[1], 12, percent_fmt)   # all coeff columns  ≈ 2 cm

    writerY.close()
    nowFunStop = datetime.now()
    print('did', filename, nowFunStop-nowFunStart)

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
    SECOND = pd.read_excel(DATA_DIR / inputFileNames[rok][1],
                          dtype={'Teryt Gminy': "Int64"})
    FIRST['Teryt Gminy']  = FIRST['Teryt Gminy'].astype(str)
    SECOND['Teryt Gminy']  = SECOND['Teryt Gminy'].astype(str)

        
    EXTRA = pd.DataFrame()

    assert_no_dupes(FIRST, [KEY1, KEY2], "FIRST")
    assert_no_dupes(SECOND, [KEY1, KEY2], "SECOND")

    X = FIRST.set_index([KEY1, KEY2])
    Y = SECOND.set_index([KEY1, KEY2])
    print("Columns in Y:", Y.columns.tolist())

    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    & (Y['Ludnosc'] >250000.0)], f"YmiastoDst250+ar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    & (Y['Ludnosc'] <=250000.0)& (Y['Ludnosc'] >100000.0)], f"YmiastoDst100-250ar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    & (Y['Ludnosc'] <=100000.0)& (Y['Ludnosc'] >40000.0)], f"YmiastoDst40-100ar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    & (Y['Ludnosc'] <=40000.0)& (Y['Ludnosc'] >20000.0)], f"YmiastoDst20-40ar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    & (Y['Ludnosc'] <=20000.0)], f"YmiastoDst20-ar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś'}) & Y['Typ obwodu'].isin({"stały"})],
                        f"YwiesDstar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)    
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                    & Y['Typ obwodu'].isin({"stały"})
                                    ], f"YmiastoDstar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
    #                                & Y['Typ obwodu'].isin({"stały"})
    #                                & (Y['Ludnosc'] <150000.0)], f"YmiastoDst150-ar{rok}.xlsx", rok,
    #                    t2absentee=True, robust=True)
    buildLinRegression (X.copy(), Y[Y['Typ obwodu'].isin({"stały"})], f"YkrajDstar{rok}.xlsx", rok,
                        t2absentee=True, robust=True)
    
    if False:
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] >250000.0)], f"YmiastoDst250+a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] <=250000.0)& (Y['Ludnosc'] >100000.0)], f"YmiastoDst100-250+a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] <=100000.0)& (Y['Ludnosc'] >40000.0)], f"YmiastoDst40-100a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] <=40000.0)& (Y['Ludnosc'] >20000.0)], f"YmiastoDst20-40a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] <=20000.0)], f"YmiastoDst20-a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś'}) & Y['Typ obwodu'].isin({"stały"})],
                            f"YwiesDsta{rok}.xlsx", rok,
                            t2absentee=True)    
        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        ], f"YmiastoDsta{rok}.xlsx", rok,
                            t2absentee=True)

        buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
                                        & Y['Typ obwodu'].isin({"stały"})
                                        & (Y['Ludnosc'] <150000.0)], f"YmiastoDst150-a{rok}.xlsx", rok,
                            t2absentee=True)
        buildLinRegression (X.copy(), Y[Y['Typ obwodu'].isin({"stały"})], f"YkrajDsta{rok}.xlsx", rok,
                            t2absentee=True)
    

    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
    #                                & Y['Typ obwodu'].isin({"stały"})
    #                                & (Y['Ludnosc'] >150000.0)], f"YmiastoDst150+{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})
    #                                & Y['Typ obwodu'].isin({"stały"})
    #                                & (Y['Ludnosc'] <150000.0)], f"YmiastoDst150-{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś', 'miasto', 'dzielnica w m.st. Warszawa'}) & Y['Typ obwodu'].isin({"stały"})], f"YkrajDst{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś'}) & Y['Typ obwodu'].isin({"stały"})], f"YwiesDst{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'}) & Y['Typ obwodu'].isin({"stały"})], f"YmiastoDst{rok}.xlsx", rok)

    ##buildLinRegression (X.copy(), Y.copy(), f"Y{rok}.xlsx", rok)
    ##buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'statek', 'zagranica'})], f"Yzagranica{rok}.xlsx", rok)
    
if __name__ == "__main__":
    main()
