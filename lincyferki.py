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
    cols_with_nan = X.columns[X.isna().any()].tolist()
    y_nan = y.isna().any()
    if not cols_with_nan and not y_nan:
        return
    mask = X[cols_with_nan].isna().any(axis=1) | y.isna()
    offenders = (
        pd.concat([X.loc[mask, cols_with_nan], y.loc[mask]], axis=1)
        .reset_index()[[KEY1, KEY2] + cols_with_nan + [tgt_name]]
    )
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
FIRST = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_utf8.xlsx")
SECOND = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx")

# compound key
KEY1, KEY2 = "Teryt Gminy", "Nr komisji"

# replace blank KEY2 with 0 and cast to int
for df in (FIRST, SECOND):
    df[KEY2] = df[KEY2].replace("", 0).astype(int)
FIRST[KEY1] = FIRST[KEY1].fillna(0).astype(int)

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

X = FIRST.set_index([KEY1, KEY2])[first_cols]
Y = SECOND.set_index([KEY1, KEY2])[targets]
#Y = SECOND.set_index([KEY1, KEY2])

# align common precincts
X, Y = X.align(Y, join="inner", axis=0)
print("Precincts analysed:", len(X))

intercepts = {}
# ------------------------------------------------------------
# 4.  OLS fits & residuals
# ------------------------------------------------------------
coefs, fits, resids = {}, {}, {}
for tgt in targets:
    nan_diagnostic(X, Y[tgt], tgt, KEY1, KEY2)
    pipe = make_pipeline(StandardScaler(with_std=False), LinearRegression())
    pipe.fit(X, Y[tgt])
    fits[tgt] = pd.Series(pipe.predict(X), index=X.index)
    resids[tgt] = Y[tgt] - fits[tgt]
    coefs[tgt] = pipe.named_steps["linearregression"].coef_
    intercepts[tgt] = pipe.named_steps["linearregression"].intercept_


# ------------------------------------------------------------
# 5.  Mahalanobis distance (old feature, unchanged)
# ------------------------------------------------------------
R = pd.concat(resids, axis=1)
inv_cov = np.linalg.inv(np.cov(R.T))
R["dist"] = R.apply(lambda r: mahalanobis(r.values, np.zeros(len(r)), inv_cov), axis=1)



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


# ------------------------------------------------------------
#  END of new block
# ------------------------------------------------------------

if False:

    # histogram for distance
    plt.figure(figsize=(35, 20))
    plt.hist(R["dist"], bins=800, alpha=0.8)
    plt.title("Mahalanobis distance")
    plt.xlabel("distance")
    plt.ylabel("precincts")
    plt.show()

# ------------------------------------------------------------
# 6.  Save top Mahalanobis outliers (old feature)
# ------------------------------------------------------------


#outliers = (
#    R["dist"].nlargest(TOP_N).reset_index()  # MultiIndex -> columns
#)
#outliers.to_excel("top_outlier_precincts.xlsx", index=False)
#print("Mahalanobis outliers written to top_outlier_precincts.xlsx")

# ============================================================
# 8A.  NEW: difference metric D  (c1 - c2) residual
# ============================================================
c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

obs_diff = Y[c1] - Y[c2]
fit_diff = fits[c1] - fits[c2]
Y["D"] = obs_diff - fit_diff            # Series on same MultiIndex

# ------------------------------------------------------------
# 8B.  Plot histogram of D & relative D
# ------------------------------------------------------------
#fig, axes = plt.subplots(1, 2, figsize=(60, 20))
plt.figure (figsize=(60, 20))

#axes[0].hist(D, bins=800, alpha=0.8, color="steelblue")
#axes[0].set(title="Histogram D = (c1−c2) − predicted", xlabel="D", ylabel="precincts")

if False:
    plt.hist(Y["D"], bins=601, alpha=0.8, color="steelblue",range=(-300, 300))
    #plt.set(title="Histogram D = (c1−c2) − predicted", xlabel="D", ylabel="precincts",xlim=(-200, 200))
    plt.title="Histogram D = (c1−c2) − predicted"
    plt.xlabel="D"
    plt.ylabel="precincts"
    plt.xlim=(-200, 200)


    plt.show()

denom   = (Y[c1] + Y[c2])**.5
D_rel   = Y["D"] / denom
D_rel   = D_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
D_rel   = D_rel.dropna()
Y["Dnorm"] = D_rel

if False:
    plt.figure (figsize=(60, 20))
    plt.hist(Y["Dnorm"], bins=800, alpha=0.8, color="indianred")
    plt.title="Histogram of D / (c1+c2)^.5"
    plt.xlabel="normalized D"
    plt.ylabel="precincts"

    plt.title="Histogram Dnorm = (c1−c2) − predicted (normalized)"
    plt.xlabel="Dnorm"
    plt.ylabel="precincts"
    #plt.xlim=(-200, 200)


    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 8C.  List ±100 outliers for D and for relative D
# ------------------------------------------------------------
GMINA = FIRST.set_index([KEY1, KEY2])["Gmina"]  # bring Gmina by MultiIndex

#def export_outliers(series, name, k=100):
#    pos = (
#        series.nlargest(k)
#        .rename("metric")
#        .reset_index()
#        .merge(GMINA.reset_index(), on=[KEY1, KEY2], how="left")
#        [[KEY1, KEY2, "Gmina", "metric"]]
#    )
#    neg = (
#        series.nsmallest(k)
#        .rename("metric")
#        .reset_index()
#        .merge(GMINA.reset_index(), on=[KEY1, KEY2], how="left")
#        [[KEY1, KEY2, "Gmina", "metric"]]
#    )
#    pos.to_excel(f"outliers_positive_{name}.xlsx", index=False)
#    neg.to_excel(f"outliers_negative_{name}.xlsx", index=False)
#    print(f"Wrote top ±{k} outliers for {name} "
#          f"to outliers_positive_{name}.xlsx / outliers_negative_{name}.xlsx")
#
#export_outliers(D, "D")
#export_outliers(D_rel, "D_rel")


writer = pd.ExcelWriter("outliers.xlsx", engine="xlsxwriter")

# ---------- helper: take any Series of scores ----------------
BASE = FIRST.set_index([KEY1, KEY2])          # full original columns
BASE2 = SECOND.set_index([KEY1, KEY2])          # full original columns
TOP_N = BASE.shape[0]//3

def sheet_name(label, sign):
    return f"{label}_{sign}"

large = Y["D"].nlargest(TOP_N)
small = Y["D"].nsmallest(TOP_N)

largeFull = large.to_frame().join(BASE2, how="left").reset_index() 
smallFull = small.to_frame().join(BASE2, how="left").reset_index() 



def add_outliers(series: pd.Series, label: str, k: int = TOP_N):
    for sign, slicer in [("pos", series.nlargest(k)),
                         ("neg", series.nsmallest(k))]:
        sheet = sheet_name(label, sign)
        df = (
            slicer.rename("metric")
                  .to_frame()
                  .join(BASE, how="left")     # keep all original cols
                  .reset_index()              # bring keys back as columns
        )
        df.to_excel(writer, sheet_name=sheet, index=False)

# ---------- 6A. Mahalanobis ----------------------------------
mah = (
    R["dist"].nlargest(TOP_N)
      .rename("metric")
      .to_frame()
      .join(BASE, how="left")
      .reset_index()
)
mah.to_excel(writer, sheet_name="Mahalanobis", index=False)

# ---------- 6B.  ±100 outliers for D and D_rel ---------------
add_outliers(Y["D"],      "D")
add_outliers(Y["Dnorm"],  "Dnorm")

# ---------- 6C.  save & finish -------------------------------
writer.close()
print(f"✓  All outlier tables written to outliers.xlsx")




# ------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------
def nan_diagnostic(X, y, tgt_name, KEY1, KEY2):
    cols_with_nan = X.columns[X.isna().any()].tolist()
    y_nan = y.isna().any()
    if not cols_with_nan and not y_nan:
        return
    mask = X[cols_with_nan].isna().any(axis=1) | y.isna()
    offenders = (
        pd.concat([X.loc[mask, cols_with_nan], y.loc[mask]], axis=1)
        .reset_index()[[KEY1, KEY2] + cols_with_nan + [tgt_name]]
    )
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

# CYFERKI

# compound key
KEY1, KEY2 = "Teryt Gminy", "Nr komisji"

use2 = [
    'Liczba niewykorzystanych kart do głosowania',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym (liczba podpisów w spisie oraz adnotacje o\xa0wydaniu karty bez potwierdzenia podpisem w\xa0spisie)',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym (łącznie)',
    'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania',
    'Liczba kart wyjętych z\xa0urny',
    'Liczba kart ważnych',
    'Liczba głosów ważnych oddanych łącznie na obu kandydatów (z\xa0kart ważnych)',
    'NAWROCKI Karol Tadeusz',
    'TRZASKOWSKI Rafał Kazimierz'
]

#use2 = [
#    'NAWROCKI Karol Tadeusz',
#    'TRZASKOWSKI Rafał Kazimierz',
#]

titles = {
    'Liczba niewykorzystanych kart do głosowania': 'karty niewykorzystane',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym (liczba podpisów w spisie oraz adnotacje o\xa0wydaniu karty bez potwierdzenia podpisem w\xa0spisie)' : 'karty wydane w lokalu',
    'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym (łącznie)' : 'karty wydane łącznie',
    'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania': 'zaświadczenia',
    'Liczba kart wyjętych z\xa0urny' : 'karty wyjęte',
    'Liczba kart ważnych' : 'karty ważne',
    'Liczba głosów ważnych oddanych łącznie na obu kandydatów (z\xa0kart ważnych)' : 'głosy ważne',
    'NAWROCKI Karol Tadeusz': 'NAWROCKI',
    'TRZASKOWSKI Rafał Kazimierz' : 'TRZASKOWSKI'

}


def cl_band(n, p_cat, p_conf=0.95):
    """Normal-approx 2-sided band for Bin(n, p_cat)."""
    z     = abs(statistics.NormalDist().inv_cdf((1 - p_conf) / 2))
    mean  = n * p_cat
    sd    = math.sqrt(n * p_cat * (1 - p_cat))
    lo    = max(0, int(math.floor(mean - z * sd)))
    hi    = min(n, int(math.ceil (mean + z * sd)))
    return lo, hi

def mean_and_ci(counts, values, n, p_conf=0.95):
    """Sample mean & normal-approx CI for a discrete variable."""
    mean = sum(v * c for v, c in zip(values, counts)) / n
    if n > 1:
        var = sum(c * (v - mean) ** 2 for v, c in zip(values, counts)) / (n - 1)
    else:
        var = 0.0
    z = abs(statistics.NormalDist().inv_cdf((1 - p_conf) / 2))
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se

def plot_histograms(
        paired_histograms,
        p_conf        = 0.95,
        category_labels=None,
        values_for_mean=None,
        band_color    = "grey",
        band_alpha    = 0.40
    ):
    """
    paired_histograms : [(title, [c0,…,c{k-1}, total]), …]
    category_labels   : list[str] (len k) – labels under bars
    values_for_mean   : list[float] (len k) – supply to compute & draw mean±CI
    """
    palette = plt.cm.tab10.colors
    for idx, (title, data) in enumerate(paired_histograms):
        counts, n = data[:-1], data[-1]
        k         = len(counts)

        labels    = category_labels if category_labels else [str(i) for i in range(k)]
        lo, hi    = cl_band(n, p_cat=1 / k, p_conf=p_conf)

        fig, ax   = plt.subplots(figsize=(8, 4))
        ax.bar(range(k), counts, color=palette[idx % len(palette)])
        ax.axhspan(lo, hi, color=band_color, alpha=band_alpha)

        ymax = max(max(counts), hi)
        for c, v in enumerate(counts):
            ax.text(c, v + 0.02*ymax, str(v), ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(k), labels)
        ax.set_ylim(0, 1.15*ymax)
        ax.set_xlabel("category")
        ax.set_ylabel("count")

        subtitle = ""
        if values_for_mean is not None and len(values_for_mean) == k:
            mean, lo_m, hi_m = mean_and_ci(counts, values_for_mean, n, p_conf)
            ax.axvline(mean, color="black", linestyle="--", lw=1.2)
            ax.axvspan(lo_m, hi_m, color="black", alpha=0.10)
            subtitle = f" | mean={mean:.2f} CI=[{lo_m:.2f}, {hi_m:.2f}]"

        ax.set_title(f"{title} | n={n}, p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle}")
        fig.tight_layout()

histograms = {}
pentagrams = {}
#for ttt, nm in [(smallFull, 'small'), (largeFull, 'large')]:
for ttt, nm in [(largeFull, 'large')]:
    for e in use2:
        s = [0]*10
        p = [0]*5
        count = 0
        for idx, row in ttt.iterrows():
            if not pd.isna(row[e]):
                v = row[e]
                if v < 50:
                    continue
                s[round(v)%10] += 1
                p[round(v)%5] += 1
                count += 1
        if 1000 <= count:
            s.append(count)
            p.append(count)
            histograms [nm + ' ' + titles[e]] = s
            pentagrams [nm + ' ' + titles[e]] = p


#haveHistograms = [e for e in use2 if e in histograms]
plot_histograms([(e, histograms[e]) for e in histograms], p_conf=0.95
                #,
                #values_for_mean = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
                )
plot_histograms([(e, pentagrams[e]) for e in pentagrams], p_conf=0.95,
                category_labels=['0 i 5','1 i 6', '2 i 7', '3 i 8', '4 i 9'])
#plot_histograms([(titles[e], histograms[e]) for e in haveHistograms], p_conf=0.97
#                #,values_for_mean = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
#                )
#plot_histograms([(titles[e], pentagrams[e]) for e in haveHistograms], p_conf=0.97,
#                category_labels=['0 i 5','1 i 6', '2 i 7', '3 i 8', '4 i 9'])
plt.show()

sys.exit(0)


def cl_bandOld(n, p_conf=0.95, p_cat=0.1):
    """Two-sided normal-approx. band for Bin(n, p_cat) at confidence p_conf."""
    z = abs(statistics.NormalDist().inv_cdf((1 - p_conf) / 2))
    mean = n * p_cat
    sd   = math.sqrt(n * p_cat * (1 - p_cat))
    lo   = max(0, int(math.floor(mean - z * sd)))
    hi   = min(n, int(math.ceil (mean + z * sd)))
    return lo, hi

def plot_histogramsOld(paired_histograms, p_conf=0.95):
    """
    paired_histograms : [(title, [c0, …, c9, n]), …]
    p_conf            : confidence level, e.g. 0.95
    """
    colours = plt.cm.tab10.colors
    for idx, (title, data) in enumerate(paired_histograms):
        counts, n   = data[:10], data[10]
        lo, hi      = cl_band(n, p_conf)
        cat_idx     = range(10)
        color       = colours[idx % len(colours)]
        
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(cat_idx, counts, color=color)
        ax.axhspan(lo, hi, color='k', alpha=0.4)          # confidence band
        
        for c, v in zip(cat_idx, counts):                  # annotate bars
            ax.text(c, v + 0.02*max(max(counts), hi), str(v),
                    ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(cat_idx)
        ax.set_xlabel("category")
        ax.set_ylabel("count")
        ax.set_ylim(0, 1.15*max(max(counts), hi))
        ax.set_title(f"{title}  |  n={n}, p={p_conf:.2f}, band=[{lo}, {hi}]")
        fig.tight_layout()
    plt.show()    

# ---------- example ---------------------------------------------------------
# Delete this block and supply your own data.
#H = [
#    [12,  9, 11, 15,  8, 10,  7,  9, 11,  8, 100],
#    [ 8, 14,  6, 12, 11, 14, 12,  7,  6, 10, 100],
#    [11, 10,  9, 10, 12,  8, 11,  7, 13,  9, 100],
#    [13,  8, 10,  9, 12, 11,  7, 10,  9, 11, 100],
#    [10, 12,  8, 11,  9, 13,  9, 12,  8,  8, 100],
#    [ 9, 10, 12,  8, 13,  9, 11,  8, 11,  9, 100],
#]





print (SECOND['Liczba kopert na kartę do głosowania w\xa0głosowaniu korespondencyjnym wrzuconych do urny'])


# replace blank KEY2 with 0 and cast to int
for df in (FIRST, SECOND):
    df[KEY2] = df[KEY2].replace("", 0).astype(int)
FIRST[KEY1] = FIRST[KEY1].fillna(0).astype(int)

assert_no_dupes(FIRST, [KEY1, KEY2], "FIRST")
assert_no_dupes(SECOND, [KEY1, KEY2], "SECOND")

columns1 = FIRST.columns.tolist()
columns2 = SECOND.columns.tolist()

print (columns1)
print (columns2)
sys.exit(0)

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
    "Liczba uprawnionych w chwili zakończenia",
    "zaświadczenia",
    "Liczba głosów nieważnych",
] + cands

targets = [
    "Liczba głosów nieważnych",
    "NAWROCKI Karol Tadeusz",          # c1
    "TRZASKOWSKI Rafał Kazimierz",     # c2
]

X = FIRST.set_index([KEY1, KEY2])[first_cols]
Y = SECOND.set_index([KEY1, KEY2])[targets]

# align common precincts
X, Y = X.align(Y, join="inner", axis=0)
print("Precincts analysed:", len(X))

intercepts = {}
# ------------------------------------------------------------
# 4.  OLS fits & residuals
# ------------------------------------------------------------
coefs, fits, resids = {}, {}, {}
for tgt in targets:
    nan_diagnostic(X, Y[tgt], tgt, KEY1, KEY2)
    pipe = make_pipeline(StandardScaler(with_std=False), LinearRegression())
    pipe.fit(X, Y[tgt])
    fits[tgt] = pd.Series(pipe.predict(X), index=X.index)
    resids[tgt] = Y[tgt] - fits[tgt]
    coefs[tgt] = pipe.named_steps["linearregression"].coef_
    intercepts[tgt] = pipe.named_steps["linearregression"].intercept_


# ------------------------------------------------------------
# 5.  Mahalanobis distance (old feature, unchanged)
# ------------------------------------------------------------
R = pd.concat(resids, axis=1)
inv_cov = np.linalg.inv(np.cov(R.T))
R["dist"] = R.apply(lambda r: mahalanobis(r.values, np.zeros(len(r)), inv_cov), axis=1)



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


# ------------------------------------------------------------
#  END of new block
# ------------------------------------------------------------



# histogram for distance
plt.figure(figsize=(35, 20))
plt.hist(R["dist"], bins=800, alpha=0.8)
plt.title("Mahalanobis distance")
plt.xlabel("distance")
plt.ylabel("precincts")
plt.show()

# ------------------------------------------------------------
# 6.  Save top Mahalanobis outliers (old feature)
# ------------------------------------------------------------
TOP_N = 100
outliers = (
    R["dist"].nlargest(TOP_N).reset_index()  # MultiIndex -> columns
)
outliers.to_excel("top_outlier_precincts.xlsx", index=False)
print("Mahalanobis outliers written to top_outlier_precincts.xlsx")

# ============================================================
# 8A.  NEW: difference metric D  (c1 - c2) residual
# ============================================================
c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

obs_diff = Y[c1] - Y[c2]
fit_diff = fits[c1] - fits[c2]
D = obs_diff - fit_diff            # Series on same MultiIndex

# ------------------------------------------------------------
# 8B.  Plot histogram of D & relative D
# ------------------------------------------------------------
#fig, axes = plt.subplots(1, 2, figsize=(60, 20))
plt.figure (figsize=(60, 20))

#axes[0].hist(D, bins=800, alpha=0.8, color="steelblue")
#axes[0].set(title="Histogram D = (c1−c2) − predicted", xlabel="D", ylabel="precincts")

plt.hist(D, bins=601, alpha=0.8, color="steelblue",range=(-300, 300))
#plt.set(title="Histogram D = (c1−c2) − predicted", xlabel="D", ylabel="precincts",xlim=(-200, 200))
plt.title="Histogram D = (c1−c2) − predicted"
plt.xlabel="D"
plt.ylabel="precincts"
plt.xlim=(-200, 200)

denom   = Y[c1] + Y[c2]
D_rel   = D / denom
D_rel   = D_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
D_rel   = D_rel.dropna()

#axes[1].hist(D_rel, bins=800, alpha=0.8, color="indianred")
#axes[1].set(title="Histogram of D / (c1+c2)", xlabel="relative D", ylabel="precincts")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8C.  List ±100 outliers for D and for relative D
# ------------------------------------------------------------
GMINA = FIRST.set_index([KEY1, KEY2])["Gmina"]  # bring Gmina by MultiIndex

def export_outliers(series, name, k=100):
    pos = (
        series.nlargest(k)
        .rename("metric")
        .reset_index()
        .merge(GMINA.reset_index(), on=[KEY1, KEY2], how="left")
        [[KEY1, KEY2, "Gmina", "metric"]]
    )
    neg = (
        series.nsmallest(k)
        .rename("metric")
        .reset_index()
        .merge(GMINA.reset_index(), on=[KEY1, KEY2], how="left")
        [[KEY1, KEY2, "Gmina", "metric"]]
    )
    pos.to_excel(f"outliers_positive_{name}.xlsx", index=False)
    neg.to_excel(f"outliers_negative_{name}.xlsx", index=False)
    print(f"Wrote top ±{k} outliers for {name} "
          f"to outliers_positive_{name}.xlsx / outliers_negative_{name}.xlsx")

export_outliers(D, "D")
export_outliers(D_rel, "D_rel")



