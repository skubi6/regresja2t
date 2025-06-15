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

nowStart = datetime.now()


PKWcolumns = [
    'Nr komisji',
    'Gmina',
    'Teryt Gminy',
    'Powiat',
    'Teryt Powiatu',
    'Województwo',
    'Siedziba',
    'Typ obwodu',
    'Typ obszaru',
    'Liczba kart do głosowania otrzymanych przez obwodową komisję wyborczą, ustalona po ich przeliczeniu przed rozpoczęciem głosowania z\xa0uwzględnieniem ewentualnych kart otrzymanych z\xa0rezerwy',
    'Liczba wyborców uprawnionych do głosowania (umieszczonych w\xa0spisie, z\xa0uwzględnieniem dodatkowych formularzy) w\xa0chwili zakończenia głosowania',
    'Liczba niewykorzystanych kart do głosowania',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym (liczba podpisów w spisie oraz adnotacje o\xa0wydaniu karty bez potwierdzenia podpisem w\xa0spisie)',
    'Liczba wyborców, którym wysłano pakiety wyborcze',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym (łącznie)',
    'Liczba wyborców głosujących przez pełnomocnika (liczba kart do głosowania wydanych na podstawie aktów pełnomocnictwa otrzymanych przez obwodową komisję wyborczą)',
    'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania',
    'Liczba otrzymanych kopert zwrotnych w\xa0głosowaniu korespondencyjnym',
    'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których nie było oświadczenia o\xa0osobistym i\xa0tajnym oddaniu głosu',
    'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których oświadczenie nie było podpisane przez wyborcę',
    'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których nie było koperty na kartę do głosowania',
    'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których znajdowała się niezaklejona koperta na kartę do głosowania',
    'Liczba kopert na kartę do głosowania w\xa0głosowaniu korespondencyjnym wrzuconych do urny',
    'Liczba kart wyjętych z\xa0urny',
    'w tym liczba kart wyjętych z\xa0kopert na kartę do głosowania w głosowaniu korespondencyjnym',
    'Liczba kart nieważnych (bez pieczęci obwodowej komisji wyborczej lub inne niż urzędowo ustalone)',
    'Liczba kart ważnych',
    'Liczba głosów nieważnych (z\xa0kart ważnych)',
    'w\xa0tym z\xa0powodu postawienia znaku „X”\xa0obok nazwiska dwóch lub większej liczby kandydatów',
    'w\xa0tym z\xa0powodu niepostawienia znaku „X”\xa0obok nazwiska żadnego kandydata',
    'w\xa0tym z\xa0powodu postawienia znaku „X”\xa0wyłącznie obok nazwiska skreślonego kandydata',
    'Liczba głosów ważnych oddanych łącznie na wszystkich kandydatów (z\xa0kart ważnych)',
    'BARTOSZEWICZ Artur',
    'BIEJAT Magdalena Agnieszka',
    'BRAUN Grzegorz Michał',
    'HOŁOWNIA Szymon Franciszek',
    'JAKUBIAK Marek',
    'MACIAK Maciej',
    'MENTZEN Sławomir Jerzy',
    'NAWROCKI Karol Tadeusz',
    'SENYSZYN Joanna',
    'STANOWSKI Krzysztof Jakub',
    'TRZASKOWSKI Rafał Kazimierz',
    'WOCH Marek Marian',
    'ZANDBERG Adrian Tadeusz'
]

# ------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------
def nan_diagnostic(X, y, tgt_name, KEY1, KEY2):
    #cols_with_nan = X.columns[X.isna().any()].tolist()
    ignore = ['Powiat', 'Teryt Powiatu', 'Województwo']
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
#FIRST = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_utf8-modif.xlsx")
#SECOND = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_w_drugiej_turze_utf8-modif.xlsx")
FIRST = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_utf8-fixed.xlsx")
FIRST.loc[FIRST['Typ obszaru'].isin(['zagranica', 'statek']), 'Teryt Gminy'] = 9999999
SECOND = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx")
SECOND.loc[SECOND['Typ obszaru'].isin(['zagranica', 'statek']), 'Teryt Gminy'] = 9999999
print (DATA_DIR / "commission_combined.xlsx")
EXTRA = pd.read_excel(DATA_DIR / "commission_combined.xlsx")
#print ('step 0')
#print (EXTRA)

EXTRA.rename(columns={"Nr obw.": "Nr komisji", "TERYT gminy" : "Teryt Gminy"}, inplace=True)

# compound key
KEY1, KEY2 = "Teryt Gminy", "Nr komisji"
print ('step 1')

print (EXTRA)

# replace blank KEY2 with 0 and cast to int
for df in (FIRST, SECOND, EXTRA):
    df[KEY2] = df[KEY2].replace("", 0).astype(int)
FIRST[KEY1] = FIRST[KEY1].fillna(0).astype(int)

SECOND_JOIN = (
    SECOND.merge(EXTRA, on=[KEY1, KEY2], how="inner",
                 suffixes=("", "_extra"))
)

print("Rows in joined SECOND+EXTRA:", len (SECOND), len(SECOND_JOIN))


assert_no_dupes(FIRST, [KEY1, KEY2], "FIRST")
assert_no_dupes(SECOND, [KEY1, KEY2], "SECOND")

# ------------------------------------------------------------
# 3.  Design & target matrices
# ------------------------------------------------------------
cands = [
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
]
first_cols = [
    #"Liczba uprawnionych w chwili zakończenia",
    'Liczba wyborców uprawnionych do głosowania (umieszczonych w\xa0spisie, z\xa0uwzględnieniem dodatkowych formularzy) w\xa0chwili zakończenia głosowania',
    #"zaświadczenia",
    'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania',
    #"Liczba głosów nieważnych",
    'Liczba głosów nieważnych (z\xa0kart ważnych)'
] + cands

targets = [
    #"Liczba głosów nieważnych",
    'Liczba głosów nieważnych (z\xa0kart ważnych)',
    "NAWROCKI Karol Tadeusz",          # c1
    "TRZASKOWSKI Rafał Kazimierz",     # c2
]

X = FIRST.set_index([KEY1, KEY2]) #[first_cols]


Y = SECOND.set_index([KEY1, KEY2]) #[targets]
print("Columns in SECOND:", SECOND.columns.tolist())
print("Columns in Y:", Y.columns.tolist())

#Y = SECOND.set_index([KEY1, KEY2])

def buildLinRegression (
        Xarg, Yarg, filename
):
    print('args received size', Xarg.shape[0], Yarg.shape[0], filename)
    nowFunStart = datetime.now()
    # align common precincts
    Xarg, Yarg = Xarg.align(Yarg, join="inner", axis=0)
    print('args ailgned  size', Xarg.shape[0], Yarg.shape[0])
    print("Precincts analysed:", len(Xarg))

    intercepts = {}
    # ------------------------------------------------------------
    # 4.  OLS fits & residuals
    # ------------------------------------------------------------
    coefs, fits, resids = {}, {}, {}
    for tgt in targets:
        nan_diagnostic(Xarg, Yarg[tgt], tgt, KEY1, KEY2)
        pipe = make_pipeline(StandardScaler(with_std=False), LinearRegression())
        pipe.fit(Xarg[first_cols], Yarg[tgt])
        Yarg['fits'+tgt] = pd.Series(pipe.predict(Xarg[first_cols]), index=Xarg.index)
        Yarg['resids'+tgt] = Yarg[tgt] - Yarg['fits'+tgt]
        coefs[tgt] = pipe.named_steps["linearregression"].coef_
        intercepts[tgt] = pipe.named_steps["linearregression"].intercept_


    # ------------------------------------------------------------
    # 5A.  >>> NEW <<<  Collect and display regression coefficients
    # ------------------------------------------------------------
    #  (Put this right after the model-fitting loop that fills `coefs`)

    # 1.  During the loop you already stored
    #       coefs[tgt]  →  NumPy array of slopes (same order as `first_cols`)
    #     Add intercepts alongside:
    #            intercepts[tgt] = pipe.named_steps["linearregression"].intercept_
    #     Make sure the loop above now records this:
    #       intercepts = {}

    # ------------------------------------------------------------
    #  BEGIN block to add just below the loop  --------------------
    # ------------------------------------------------------------

    predictor_names = ["Intercept"] + first_cols

    coef_tbl = pd.DataFrame(index=predictor_names)

    for tgt in targets:
        coef_tbl[tgt] = [intercepts[tgt]] + coefs[tgt].tolist()

    # convert to percent & round to 4 decimal places
    coef_tbl = (coef_tbl * 100).round(2)

    # custom float formatter for aligned output
    float_fmt = lambda x: f"{x:10.2f}"

    print("\n================ Linear-Hypothesis Coefficients (% units) ================")
    print(coef_tbl.to_string(float_format=float_fmt))
    print("==========================================================================\n")


    c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

    Yarg["obs_diff"] = Yarg[c1] - Yarg[c2]
    Yarg["fit_diff"] = Yarg['fits' + c1] - Yarg['fits' + c2]
    Yarg["D"] = Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex
    Yarg["Drev"] = -Yarg["obs_diff"] - Yarg["fit_diff"]            # Series on same MultiIndex



    denom   = (Yarg[c1] + Yarg[c2])**.5
    D_rel   = Yarg["D"] / denom
    D_rel   = D_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    D_rel   = D_rel.dropna()
    Yarg["Dnorm"] = D_rel
    

    Drev_rel   = Yarg["D"] / denom
    Drev_rel   = Drev_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
    Drev_rel   = Drev_rel.dropna()
    Yarg["Drevnorm"] = D_rel

    nowAfterInit = datetime.now()

    Yarg.reset_index(inplace=True)

    writerY = pd.ExcelWriter(filename, engine="xlsxwriter")
    Yarg.to_excel(writerY, sheet_name="Yarg", index=False)

    writerY.close()
    nowFunStop = datetime.now()
    print('did', filename, nowFunStop-nowFunStart)

buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'miasto', 'dzielnica w m.st. Warszawa'})], "Ymiasta.xlsx")
buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'wieś', 'miasto i wieś'})], "Ywies.xlsx")
buildLinRegression (X.copy(), Y[Y['Typ obszaru'].isin({'statek', 'zagranica'})], "Yzagranica.xlsx")
buildLinRegression (X.copy(), Y.copy(), "Y.xlsx")


#print (nowAfterInit-nowStart)

yyyy = ['Gmina', 'Powiat', 'Teryt Powiatu', 'Województwo', 'Siedziba', 'Typ obwodu', 'Typ obszaru', 'Liczba kart do głosowania otrzymanych przez obwodową komisję wyborczą, ustalona po ich przeliczeniu przed rozpoczęciem głosowania z\xa0uwzględnieniem ewentualnych kart otrzymanych z\xa0rezerwy', 'Liczba wyborców uprawnionych do głosowania (umieszczonych w\xa0spisie, z\xa0uwzględnieniem dodatkowych formularzy) w\xa0chwili zakończenia głosowania', 'Liczba niewykorzystanych kart do głosowania', 'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym (liczba podpisów w spisie oraz adnotacje o\xa0wydaniu karty bez potwierdzenia podpisem w\xa0spisie)', 'Liczba wyborców, którym wysłano pakiety wyborcze', 'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym (łącznie)', 'Liczba wyborców głosujących przez pełnomocnika (liczba kart do głosowania wydanych na podstawie aktów pełnomocnictwa otrzymanych przez obwodową komisję wyborczą)', 'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania', 'Liczba otrzymanych kopert zwrotnych w\xa0głosowaniu korespondencyjnym', 'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których nie było oświadczenia o\xa0osobistym i\xa0tajnym oddaniu głosu', 'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których oświadczenie nie było podpisane przez wyborcę', 'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których nie było koperty na kartę do głosowania', 'Liczba kopert zwrotnych w\xa0głosowaniu korespondencyjnym, w\xa0których znajdowała się niezaklejona koperta na kartę do głosowania', 'Liczba kopert na kartę do głosowania w\xa0głosowaniu korespondencyjnym wrzuconych do urny', 'Liczba kart wyjętych z\xa0urny', 'w tym liczba kart wyjętych z\xa0kopert na kartę do głosowania w głosowaniu korespondencyjnym', 'Liczba kart nieważnych (bez pieczęci obwodowej komisji wyborczej lub inne niż urzędowo ustalone)', 'Liczba kart ważnych', 'Liczba głosów nieważnych (z\xa0kart ważnych)', 'w\xa0tym z\xa0powodu postawienia znaku „X”\xa0obok nazwisk obu kandydatów', 'w\xa0tym z\xa0powodu niepostawienia znaku „X”\xa0obok nazwiska żadnego kandydata', 'Liczba głosów ważnych oddanych łącznie na obu kandydatów (z\xa0kart ważnych)', 'NAWROCKI Karol Tadeusz', 'TRZASKOWSKI Rafał Kazimierz']
