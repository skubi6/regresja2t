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
FIRST = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_utf8-modif.xlsx")
SECOND = pd.read_excel(DATA_DIR / "protokoly_po_obwodach_w_drugiej_turze_utf8-modif.xlsx")

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
fig, axes = plt.subplots(1, 2, figsize=(60, 20))

axes[0].hist(D, bins=800, alpha=0.8, color="steelblue")
axes[0].set(title="Histogram of D = (c1−c2) − predicted", xlabel="D", ylabel="precincts")

denom   = Y[c1] + Y[c2]
D_rel   = D / denom
D_rel   = D_rel.replace([np.inf, -np.inf], np.nan)   # guard against div-by-0
D_rel   = D_rel.dropna()

axes[1].hist(D_rel, bins=800, alpha=0.8, color="indianred")
axes[1].set(title="Histogram of D / (c1+c2)", xlabel="relative D", ylabel="precincts")

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

