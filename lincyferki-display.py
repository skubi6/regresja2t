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


pis = {
    'Marka Jakubiaka',
    'Jakubiaka',
    'Krzysztofa Jakuba Stanowskiego',
    'Romualda Starosielca',
    'Wiesława Lewickiego',
    'Eugeniusza Maciejewskiego',
    'Karola Nawrockiego',
    'Sławomira Jerzego Mentzena',
    'Adama Nawary',
    'Adama Nawary, Pawła Tanajno',
    'Aldony Anny Skirgiełło',
    'Andrzeja Jana Kasela',
    'Artura Bartoszewicza',
    'Dawida Bohdana Jackiewicza',
    'Dominiki Jasińskiej',
    'Eugeniusza Maciejewskiego',
    'Grzegorza Kołek',
    'Grzegorza Michała Bra',
    'Grzegorza Michała Brauna',
    'Grzegorza Michała Bra, Grzegorza Michała Brauna',
    'Jakuba Perkowskiego',
    'Jana Wojciecha Kubania'
}

pis = {
       'Adama Nawary',
       'Adama Nawary, Pawła Tanajno',
       'Adriana Zandberga',
       'Aj',
       'Aldony Anny Skirgiełło',
       'Andrzeja Jana Kasela',
       'Artura Bartoszewicza',
       'Dawida Bohdana Jackiewicza',
       'Dominiki Jasińskiej',
       'Eugeniusza Maciejewskiego',
       'Grzegorza Kołek',
       'Grzegorza Michała Bra',
       'Grzegorza Michała Bra, Grzegorza Michała Brauna',
       'Grzegorza Michała Brauna',
       'Grzegorza Michała Brauna, Grzegorza Michała Bra',
       'Jakuba Perkowskiego',
       'Jakubiaka',
       'Jana Wojciecha Kubania',
       
       'Jolanty Dudy',
       'Kajłola Nawrockiego',
       'Kajłola Nawrockiego, Karola Nawrockiego',
       'Karola Nawrockiego',
       'Katarzyny Anny Łysik',
       'Katarzyny Cichos',
       'Krzysztofa Andrzeja Sitko',
       'Krzysztofa Jakuba Stanowskiego',
       'Krzysztofa Tołwińskiego',
       'Macieja Maciaka',
       
       'Marcina Bugajskiego',
       'Marka Jakubiaka',
       'Marka Wocha',
       'Marty Ratuszyńskiej',
       'Pawła Tanajno',
       'Piotra Daniela Lechowicza',
       'Piotra Szumlewicza',
       'Roberta Śledzia',
       'Roberta Więcko',
       'Romltalda Starosielca',
       'Romualda Starosielca',
       'Sebastiana Rossa',
       'Si',
       'Sławomira Jerzego Mentzena',
       'Stanisława Żółtka',
       'Tomasza Ziółkowskiego',
       'Wiesława Lewickiego',
       'Włodzimierza Rynkowskiego',
       'Wocha',
       'Wocha, Marka Wocha',
       'Wojciecha Papis',
       'Zbigniewa Litke'

    }

antypis = {
    'Magdaleny Biejat',
    'Magdaleny Biej At',
    'Magdaleny Biej At, Magdaleny Biejat',
    'Rafała Trzaskowskiego',
    'Rafai',
    'Rafaj, Rafała Trzaskowskiego',
    'Szymona Hołowni',
    'Szymona Hoi',
    'Adriana Zandberga',
    'Joanny Senyszyn',    
    
}

znaneKomitety = pis | antypis

def classify (row):
    countPis = 0
    a = row["member1_candidate"]
    b = row["member2_candidate"]
    if a in pis:
        countPis += 1
    if b in pis:
        countPis += 1
    if a in antypis:
        countPis -= 1
    if b in antypis:
        countPis -= 1

    if a != '' and a != np.NaN and a not in znaneKomitety:
        print ('Nieznany komitet', a)

    if b != '' and a != np.NaN and b not in znaneKomitety:
        print ('Nieznany komitet', b)

    if 0 <countPis:
        return  "#CC0000"  # "red"
    elif countPis < 0:
        return  "#0066CC"  # "blue"
    else:
        return "black"

nowStart = datetime.now()

KEY1, KEY2 = "Teryt Gminy", "Nr komisji"
c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

DATA_DIR = Path(".")
Y = pd.read_excel(DATA_DIR / "Ymerged.xlsx")
Y["color"] = Y.apply(classify, axis=1)

nowAfterInit = datetime.now()
print (nowAfterInit-nowStart)

nowRead = datetime.now()

print ('reading time', nowRead-nowStart)
denom = (Y[c1] + Y[c2]) ** 0.5
obs_norm  = Y["obs_diff"]  / denom
fit_norm  = Y["fit_diff"]  / denom

fig_raw, ax_raw = plt.subplots(figsize=(38.4, 21.6), dpi=100)
ax_raw.scatter(
    Y["fit_diff"],
    Y["obs_diff"],
    s=4, marker=".", c=Y["color"], alpha=0.8
)
ax_raw.axvline(0, color="grey", linewidth=0.8)
ax_raw.axhline(0, color="grey", linewidth=0.8)
ax_raw.set_xlabel("fit_diff")
ax_raw.set_ylabel("obs_diff")
ax_raw.set_title("obs_diff vs fit_diff")
fig_raw.tight_layout()

# ---------- Window 2 : normalised --------------------------------------------
fig_norm, ax_norm = plt.subplots(figsize=(38.4, 21.6), dpi=100)
ax_norm.scatter(
    fit_norm,
    obs_norm,
    s=4, marker=".", color=Y["color"], alpha=0.8
)
ax_norm.axvline(0, color="grey", linewidth=0.8)
ax_norm.axhline(0, color="grey", linewidth=0.8)
ax_norm.set_xlabel("fit_diff (norm)")
ax_norm.set_ylabel("obs_diff (norm)")
ax_norm.set_title("Normalised obs_diff vs fit_diff")
fig_norm.tight_layout()

fig_D, ax_D = plt.subplots(figsize=(38.4, 21.6), dpi=100)
ax_D.scatter(
    Y["fit_diff"],
    Y["D"],
    s=6, marker=".", color=Y["color"], alpha=0.8
)
ax_D.axvline(0, color="grey", linewidth=0.8)
ax_D.axhline(0, color="grey", linewidth=0.8)


xmin, xmax = ax_D.get_xlim()
x_vals = np.array([xmin, xmax])
ax_D.plot(x_vals,  x_vals,  color="grey", linewidth=0.8)  # X = Y
ax_D.plot(x_vals, -x_vals,  color="grey", linewidth=0.8)  # X = -Y

ax_D.set_ylim(-700, 700)   # ← pick your ymin,ymax here

ax_D.set_xlabel("fit_diff")
ax_D.set_ylabel("D")
ax_D.set_title("obs_diff vs fit_diff")
fig_D.tight_layout()

# ---------- Window 2 : normalised --------------------------------------------
fig_Dnorm, ax_Dnorm = plt.subplots(figsize=(38.4, 21.6), dpi=100)
ax_Dnorm.scatter(
    fit_norm,
    Y["Dnorm"],
    s=6, marker=".", color=Y["color"], alpha=0.8
)
ax_Dnorm.axvline(0, color="grey", linewidth=0.8)
ax_Dnorm.axhline(0, color="grey", linewidth=0.8)

xmin, xmax = ax_Dnorm.get_xlim()
x_vals = np.array([xmin, xmax])
ax_Dnorm.plot(x_vals,  x_vals,  color="grey", linewidth=0.8)  # X = Y
ax_Dnorm.plot(x_vals, -x_vals,  color="grey", linewidth=0.8)  # X = -Y
ax_Dnorm.set_xlim(xmin, xmax)
ax_Dnorm.set_ylim(-12, 12)   # ← pick your ymin,ymax here

ax_Dnorm.set_xlabel("fit_diff (norm)")
ax_Dnorm.set_ylabel("D (norm)")
ax_Dnorm.set_title("Normalised obs_diff vs fit_diff")
fig_Dnorm.tight_layout()


#fig_scatter = plt.figure(figsize=(38.4, 21.6), dpi=100)

#ax = fig_scatter.add_subplot(1, 1, 1)
#ax.scatter(
#    Y["fit_diff"],
#    Y["obs_diff"],
#    s=4,              # 2 px × 2 px marker at 100 dpi  (area in pt²)
#    marker=".",       # fast, solid dot
#    color="black",
#    alpha=0.8
#)
#ax.axvline(0, color="grey", linewidth=0.8)   # fit_diff = 0
#ax.axhline(0, color="grey", linewidth=0.8)   # obs_diff = 0
#ax.set_xlabel("fit_diff")
#ax.set_ylabel("obs_diff")
#ax.set_title("obs_diff as a function of fit_diff")
#
#fig_scatter.tight_layout()
plt.show()

sys.exit(0)


fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, figsize=(60, 20), constrained_layout=True
)


# ── left-hand histogram: raw D ────────────────────────────────────────────────
ax1.hist(
    Y["D"], bins=601, alpha=0.8, color="steelblue", range=(-300, 300)
)
ax1.axvline(0, color="black", linewidth=0.8)          # central line at 0
ax1.set_title("Histogram D = (c1−c2) − predicted")
ax1.set_xlabel("D")
ax1.set_ylabel("precincts")
ax1.set_xlim(-200, 200)

# ── right-hand histogram: normalised D ────────────────────────────────────────
ax2.hist(
    Y["Dnorm"], bins=800, alpha=0.8, color="indianred", range=(-10, 10)
)
ax2.axvline(0, color="black", linewidth=0.8)          # central line at 0
ax2.set_title("Histogram Dnorm = (c1−c2) − predicted (normalized)")
ax2.set_xlabel("normalized D")
ax2.set_ylabel("precincts")
ax2.set_xlim(-10, 10)

plt.show()

nowCalc = datetime.now()

# ------------------------------------------------------------
# 8C.  List ±100 outliers for D and for relative D
# ------------------------------------------------------------

writer = pd.ExcelWriter("outliers.xlsx", engine="xlsxwriter")

# ---------- helper: take any Series of scores ----------------
TOP_N = Y.shape[0]//3

def sheet_name(label, sign):
    return f"{label}_{sign}"

criterion = "Dnorm"

large = Y.nlargest(TOP_N, criterion)
small = Y.nsmallest(TOP_N, criterion)
mid = Y.nsmallest(TOP_N*2, criterion).nlargest(TOP_N, criterion)

def add_outliers(series: pd.Series, label: str, k: int = TOP_N):
    for sign, slicer in [("pos", series.nlargest(k)),
                         ("neg", series.nsmallest(k))]:
        sheet = sheet_name(label, sign)
        df = (
            slicer.rename("metric")
                  .to_frame()
                  .join(Y, how="left")     # keep all original cols
                  .reset_index()              # bring keys back as columns
        )
        df.to_excel(writer, sheet_name=sheet, index=False)

# ---------- 6B.  ±100 outliers for D and D_rel ---------------
add_outliers(Y["D"],      "D")
add_outliers(Y["Dnorm"],  "Dnorm")

# ---------- 6C.  save & finish -------------------------------
writer.close()
print(f"✓  All outlier tables written to outliers.xlsx")

nowCalcEnd = datetime.now()

print ('outliers time', nowCalcEnd-nowCalc)


# CYFERKI

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
#for ttt, nm in [(small, 'small'), (large, 'large')]:
for ttt, nm in [(large, 'large')]:
#for ttt, nm in [(small, 'small'), (mid, 'mid'), (large, 'large')]:
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
                )
plot_histograms([(e, pentagrams[e]) for e in pentagrams], p_conf=0.95,
                category_labels=['0 i 5','1 i 6', '2 i 7', '3 i 8', '4 i 9'])

plt.show()
