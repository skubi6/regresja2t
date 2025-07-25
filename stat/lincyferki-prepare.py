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

nowStart = datetime.now()

inputFileNames = {
    2025: ("protokoly_po_obwodach_utf8-fixed.xlsx",
           "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx",
           "commission_combined.xlsx"),
    2020: ("2020-t1-wyniki_gl_na_kand_po_obwodach_utf8.xlsx",
           "2020-t2-wyniki_gl_na_kand_po_obwodach_utf8.xlsx",
           None),
    2015: ("2015-wyniki_tura1-1.xls", "2015-wyniki_tura2.xls", None)
}

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

def nan_diagnostic(X, y, tgt_name, KEY1, KEY2):
    #cols_with_nan = X.columns[X.isna().any()].tolist()
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
    print(f"\n!!! NaNs in target '{tgt_name}':")
    for k in keys:
        print("  -", k)
    offenders.to_excel(f"nan_offenders_{tgt_name}.xlsx", index=False)
    sys.exit(1)

def assert_no_dupes(df, key_cols, label):
    dup = df.duplicated(key_cols, keep=False)
    if dup.any():
        raise SystemExit(
            f"\n!!! DUPLICATE KEYS in {label}:\n"
            f"{df.loc[dup, key_cols].to_string(index=False)}\n"
        )


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
        "NAWROCKI Karol Tadeusz",          # c1
        "TRZASKOWSKI Rafał Kazimierz",     # c2
    ],
        
    2020: [
        'Liczba głosów nieważnych',
        "Andrzej Sebastian DUDA",          # c1
        "Rafał Kazimierz TRZASKOWSKI",     # c2
    ]
}
        

#Y = SECOND.set_index([KEY1, KEY2])

def buildLinRegression (
        Xarg, Yarg, filename, rok
):
    nowFunStart = datetime.now()
    # align common precincts
    Xarg, Yarg = Xarg.align(Yarg, join="inner", axis=0)

    intercepts = {}
    coefs, fits, resids = {}, {}, {}

    ALPHAS = np.logspace(-3, 3, 13)      # 1e-3 … 1e3

    for tgt in targets[rok]:
        nan_diagnostic(Xarg, Yarg[tgt], tgt, KEY1, KEY2)

        pipe = make_pipeline(
            SimpleImputer(strategy="median"),    # 1. imputacja braków
            StandardScaler(),                    # 2. pełna standaryzacja
            RidgeCV(alphas=ALPHAS, cv=5)         # 3. regresja z CV po alfach
        )

        try:
            pipe.fit(Xarg[first_cols[rok]], Yarg[tgt])
        except Exception:
            writerD = pd.ExcelWriter("debug.xlsx", engine="xlsxwriter")
            Xarg.to_excel(writerD, sheet_name="Xarg", index=True)
            Yarg[[tgt]].to_excel(writerD, sheet_name="Yarg", index=True)
            writerD.close()
            raise

        print("doing", tgt)
        # -------- prognozy ------------------------------------------------------
        Yarg["fits" + tgt] = pd.Series(
            pipe.predict(Xarg[first_cols[rok]]),
            index=Xarg.index
        )
        print(Yarg["fits" + tgt])
        Yarg["resids" + tgt] = Yarg[tgt] - Yarg["fits" + tgt]

        # -------- współczynniki w ORYGINALNEJ skali cech ------------------------
        scaler = pipe.named_steps["standardscaler"]
        ridge  = pipe.named_steps["ridgecv"]

        beta_std = ridge.coef_                       # współczynniki po skalowaniu
        beta_orig = beta_std / scaler.scale_         # „od-standaryzowanie”

        intercept_orig = (
            ridge.intercept_
            - np.sum(scaler.mean_ * beta_orig)       # korekta interceptu
        )

        coefs[tgt]      = beta_orig
        intercepts[tgt] = intercept_orig

    # tabela współczynników
    predictor_names = ["Intercept"] + first_cols[rok]
    coef_tbl = pd.DataFrame(index=predictor_names)

    for tgt in targets[rok]:
        coef_tbl[tgt] = [intercepts[tgt]] + coefs[tgt].tolist()

    # convert to percent & round to 4 decimal places
    coef_tbl = (coef_tbl * 100).round(2)

    # custom float formatter for aligned output
    float_fmt = lambda x: f"{x:10.2f}"

    print("\n================ Linear-Hypothesis Coefficients (% units) ================")
    print(coef_tbl.to_string(float_format=float_fmt))
    print("==========================================================================\n")

    c1, c2 = c[rok][0], c[rok][1]

    Yarg["obs_diff"] = Yarg[c1] - Yarg[c2]
    Yarg["fit_diff"] = Yarg['fits' + c1] - Yarg['fits' + c2]
    Yarg["D"] = Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex
    Yarg["Drev"] = -Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex


    N   = Xarg[first_cols[rok][0]]
    p   = Yarg["fits" + c1] / N
    q   = Yarg["fits" + c2] / N

    Yarg["p"] = p
    Yarg["q"] = q

    # --- parametry rozkładu różnicy Nawrocki-Trzaskowski ------------------------------------------
    mu   = N * (p - q)                                # E[A-B]
    var  = N * (p + q - (p - q)**2)                   # Var[A-B]
    sigma = np.sqrt(var)                              # odchylenie std.

    Yarg["diff_mu"]   = mu
    Yarg["diff_var"]  = var
    Yarg["diff_std"]  = sigma

    # --- 95 % przedział ufności ---------------------------------------------------

    conf = 0.95
    z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    Yarg["diff_ci95_low"]  = mu - z * sigma
    Yarg["diff_ci95_high"] = mu + z * sigma

    conf = 0.995
    z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    Yarg["diff_ci995_low"]  = mu - z * sigma
    Yarg["diff_ci995_high"] = mu + z * sigma

    conf = 0.9995
    z     = norm.ppf(0.5 + conf/2)                    # ≈ 1.95996
    Yarg["diff_ci9995_low"]  = mu - z * sigma
    Yarg["diff_ci9995_high"] = mu + z * sigma

    # --- jaki poziom ufności miałby przedział z brzegiem w obserwowanym D --------
    z_edge           = (Yarg["obs_diff"] - mu).abs() / sigma
    Yarg["x_edge"]   = 2 * norm.cdf(z_edge) - 1       # x ∈ (0, 1)

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
    
    nowAfterInit = datetime.now()

    Yarg.reset_index(inplace=True)

    writerY = pd.ExcelWriter(filename, engine="xlsxwriter")
    Yarg.to_excel(writerY, sheet_name="Yarg", index=False)

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

    FIRST = pd.read_excel(DATA_DIR / inputFileNames[rok][0])
    FIRST.loc[FIRST['Typ obszaru'].isin(['zagranica', 'statek']), terytGminy[rok]] = 9999999
    SECOND = pd.read_excel(DATA_DIR / inputFileNames[rok][1])
    SECOND.loc[SECOND['Typ obszaru'].isin(['zagranica', 'statek']), terytGminy[rok]] = 9999999
    global KEY1
    global KEY2
    KEY1, KEY2 = terytGminy[rok], nrKomisji[rok]
    for df in (FIRST, SECOND):
        df[KEY2] = df[KEY2].replace("", 0).astype(int)
    FIRST[KEY1] = FIRST[KEY1].fillna(0).astype(int)

    print ('step 1')

    assert_no_dupes(FIRST, [KEY1, KEY2], "FIRST")
    assert_no_dupes(SECOND, [KEY1, KEY2], "SECOND")

    X = FIRST.set_index([KEY1, KEY2])

    Y = SECOND.set_index([KEY1, KEY2])

    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})], f"Ymiasta{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś'})], f"Ywies{rok}.xlsx", rok)
    buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś', 'miasto', 'dzielnica w m.st. Warszawa'})], f"YkrajB{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y.copy(), f"Y{rok}.xlsx", rok)
    #buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'statek', 'zagranica'})], f"Yzagranica{rok}.xlsx", rok)
    
if __name__ == "__main__":
    main()
