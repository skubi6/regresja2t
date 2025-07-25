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
palette = plt.cm.tab10.colors

pisOld = {
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
    'Aldony Anny Skirgiełło, Aldony Anny Skirgiełi',
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
    'Rafała Trzaskowskiego, Rafai',
    
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

    if a != '' and not pd.isna (a) and a not in znaneKomitety:
        print ('Nieznany komitet', a)

    if b != '' and not pd.isna(b) and b not in znaneKomitety:
        print ('Nieznany komitet', b)

    if 0 <countPis:
        return  "red" # "#CC0000"  # "red"
    elif countPis < 0:
        return  "blue" # "#0066CC"  # "blue"
    else:
        return "black"

colors = {"red": "#CC0000", "blue": "#0066CC", "black" : "black"}
    
def squash(Y, Ylim, K):
    """
    For -Ylim <= Y <= Ylim: returns Y.
    For Y > Ylim:       returns Ylim + K*(Y - Ylim).
    For Y < -Ylim:      returns -Ylim + K*(Y + Ylim).
    """
    return np.where(
        Y >  Ylim,  Ylim + K*(Y -  Ylim),
    np.where(
        Y < -Ylim, -Ylim + K*(Y + Ylim),
                  Y
    ))


def squash_line_segments(ax, Ylimit, K, sign=+1, **plot_kw):
    """
    Draw the squashed diagonal on *ax*.

    Parameters
    ----------
    ax        : matplotlib Axes in which the scatter already lives
    Ylimit    : vertical cut-off (positive scalar)
    K         : compression factor (0 < K < 1)
    sign      : +1 → line for Y =  X
                -1 → line for Y = –X
    **plot_kw : forwarded to ax.plot (e.g. color, linewidth)
    """
    xmin, xmax = ax.get_xlim()

    # convenience aliases
    s = sign                  #  +1 or -1
    L = Ylimit

    # ---- middle segment (slope ±1, exists where |x| ≤ L) ----
    mid_x0 = max(xmin, -L)
    mid_x1 = min(xmax,  L)
    if mid_x0 < mid_x1:                      # segment visible?
        ax.plot([mid_x0, mid_x1],
                [s*mid_x0, s*mid_x1],
                **plot_kw)

    if xmax >  L:
        x0, x1 =  L, xmax
        y0_raw, y1_raw = s*x0, s*x1            #  y0_raw is ±L
        # y0 remains ±L, y1 gets squashed
        y0 =  s*L
        y1 = (-L + K*(y1_raw + L)) if y1_raw < -L else (L + K*(y1_raw - L))
        ax.plot([x0, x1], [y0, y1], **plot_kw)

    # ---- left outer segment (x < –L) ----
    if xmin < -L:
        #x0, x1 = -L, xmin
        # raw y = s*x ; here raw-y is outside the band, so squash:
        #   if raw-y >  L :   y =  L + K*(raw-y - L)
        #   if raw-y < –L :   y = -L + K*(raw-y + L)
        #y0_raw, y1_raw = s*x0, s*x1
        #y0 = -s*L
        #y1 = (-L + K*(y1_raw + L)) if y1_raw < -L else (L + K*(y1_raw - L))
        #y1 = (L + K*(y0_raw - L)) if y0_raw > L else (-L + K*(y0_raw + L))
        ax.plot([-L, xmin], [-s*L, s*((xmin+L)*K-L)], **plot_kw)

    # ---- right outer segment (x >  L) ----


Ylimit_D = 300    # ← your chosen limit for D
K_D      = 0.2    # ← your chosen squeeze factor

Ylimit_Dnorm = 5    # ← your chosen limit for D
K_Dnorm      = 0.2    # ← your chosen squeeze factor

nowStart = datetime.now()

terytGminy = {
    2025: "Teryt Gminy",
    2020: "Kod TERYT"
}

nrKomisji = {
    2025: "Nr komisji",
    2020: "Numer obwodu"
}

#KEY1, KEY2 = "Teryt Gminy", "Nr komisji"
c = {
        2025 : ["NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"],
        2020 : ["Andrzej Sebastian DUDA", "Rafał Kazimierz TRZASKOWSKI"]
}

#c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

DATA_DIR = Path(".")

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

def _draw_single_hist(ax, *,
                      counts, n, labels,
                      title, p_conf,
                      bar_colour,
                      band_color="grey", band_alpha=0.4,
                      values_for_mean=None,
                      lVisible):
    """
    Draw a single bar-histogram on the Axes 'ax'.
    The arguments are exactly what your original inner code expected.
    """
    k        = len(counts)
    lo, hi   = cl_band(n, p_cat=1 / k, p_conf=p_conf)
    palette  = plt.cm.tab10.colors

    # bars + confidence band
    ax.bar(range(k), counts, color=bar_colour)
    ax.axhspan(lo, hi, color=band_color, alpha=band_alpha)

    # labels over bars
    ymax = max(max(counts), hi)
    for c, v in enumerate(counts):
        ax.text(c, v + 0.02 * ymax, str(v),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(k), labels)
    ax.set_ylim(0, 1.15 * ymax)
    ax.set_ylabel("count")
    ax.set_title(f"{title} | n={n} p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle} {lVisible}")

    # optional mean ± CI marker
    if values_for_mean is not None and len(values_for_mean) == k:
        mean, lo_m, hi_m = mean_and_ci(counts, values_for_mean, n, p_conf)
        ax.axvline(mean, color="black", linestyle="--", lw=1.2)
        ax.axvspan(lo_m, hi_m, color="black", alpha=0.10)

# ---------------------------------------------------------------------
# --- 2.  draw a PAIR (histogram + pentagram) in one figure ----------
# ---------------------------------------------------------------------
def plot_histogram_pair(title,
                        histo_data,        # list[ counts… , total ]
                        penta_data,        # list[ counts… , total ]
                        lVisible,
                        p_conf,
                        bar_colour):
    """
    Draw the (10-bin) histogram and the (5-bin) pentagram
    for the same variable one below the other.
    """
    # unpack data ------------------------------------------------------
    h_counts, h_n = histo_data[:-1], histo_data[-1]
    p_counts, p_n = penta_data[:-1], penta_data[-1]

    # x-tick labels
    h_labels = [str(i) for i in range(10)]
    p_labels = ['0 i 5', '1 i 6', '2 i 7', '3 i 8', '4 i 9']

    # create the stacked axes -----------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, sharex=False, figsize=(8, 8))

    # top: full histogram (10 bins) -----------------------------------
    _draw_single_hist(
        ax_top,
        counts=h_counts,
        n=h_n,
        labels=h_labels,
        title=f"{title} 10-bins | n={h_n} {lVisible}",
        p_conf=p_conf,
        bar_colour=bar_colour,
        lVisible=lVisible
    )

    # bottom: pentagram (5 bins) --------------------------------------
    _draw_single_hist(
        ax_bot,
        counts=p_counts,
        n=p_n,
        labels=p_labels,
        title=f"{title} | n={p_n} p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle} {lVisible}",
        p_conf=p_conf,
        bar_colour=bar_colour,
        lVisible=lVisible
    )

    fig.tight_layout()
    return fig            # return so caller can .savefig() if desired





def plot_histogramsOld(
        paired_histograms,
        lVisible,
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

        ax.set_title(f"{title} | n={n}, p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle} {lVisible}")
        fig.tight_layout()

def displaySomething (l, *, histogramy, cyferki, rok, mergedInfix):
    lVisible = l or 'wszystko'
    Y = pd.read_excel(DATA_DIR / f"Y{l}B{rok}{mergedInfix}.xlsx")
    if mergedInfix:
        Y["class"] = Y.apply(classify, axis=1)
    else:
        Y["class"] = "black";
        #Y.loc[:, "class"] = "black"
    Y["color"] = Y["class"].map(colors)
    
    nowAfterInit = datetime.now()
    #print (nowAfterInit-nowStart)

    nowRead = datetime.now()

    #print ('reading time', nowRead-nowStart)
    denom = (Y[c[rok][0]] + Y[c[rok][1]]) ** 0.5
    obs_norm  = Y["obs_diff"]  / denom
    fit_norm  = Y["fit_diff"]  / denom

    if False:
        fig_raw, ax_raw = plt.subplots(figsize=(38.4, 21.6), dpi=100)
        ax_raw.scatter(
            Y["fit_diff"],
            Y["obs_diff"],
            s=8, marker=".", c=Y["color"], alpha=0.8
        )
        ax_raw.axvline(0, color="grey", linewidth=0.8)
        ax_raw.axhline(0, color="grey", linewidth=0.8)
        ax_raw.set_xlabel("fit_diff")
        ax_raw.set_ylabel("obs_diff")
        ax_raw.set_title("obs_diff vs fit_diff " + lVisible)
        fig_raw.tight_layout()

        # ---------- Window 2 : normalised --------------------------------------------
        fig_norm, ax_norm = plt.subplots(figsize=(38.4, 21.6), dpi=100)
        ax_norm.scatter(
            fit_norm,
            obs_norm,
            s=6, marker=".", color=Y["color"], alpha=0.8
        )
        ax_norm.axvline(0, color="grey", linewidth=0.8)
        ax_norm.axhline(0, color="grey", linewidth=0.8)

        ax_Dnorm.axhline( Ylimit_Dnorm, color="grey", linewidth=0.8)
        ax_Dnorm.axhline(-Ylimit_Dnorm, color="grey", linewidth=0.8)

        ax_norm.set_xlabel("fit_diff (norm)")
        ax_norm.set_ylabel("obs_diff (norm)")
        ax_norm.set_title("Normalised obs_diff vs fit_diff " + lVisible)
        fig_norm.tight_layout()

    if histogramy:
        fig_D, ax_D = plt.subplots(figsize=(38.4, 21.6), dpi=100)
        D_trans = squash(Y["D"], Ylimit_D, K_D)
        ax_D.scatter(
            Y["fit_diff"],
            D_trans,
            s=8, marker=".", color=Y["color"], alpha=0.8
        )


        xmin, xmax = ax_D.get_xlim()
        x_vals = np.array([xmin, xmax])
        #ax_D.plot(x_vals,  x_vals,  color="grey", linewidth=0.8)  # X = Y
        #ax_D.plot(x_vals, -x_vals,  color="grey", linewidth=0.8)  # X = -Y

        ax_D.set_ylim(-Ylimit_D*1.3, Ylimit_D*1.3)   # ← pick your ymin,ymax here

        ax_D.axvline(0, color="grey", linewidth=0.8)
        ax_D.axhline(0, color="grey", linewidth=0.8)

        ax_D.axhline( Ylimit_D, color="grey", linewidth=0.8)
        ax_D.axhline(-Ylimit_D, color="grey", linewidth=0.8)

        squash_line_segments(ax_D,     Ylimit_D,     K_D,  sign=+1,
                             color="grey", linewidth=0.8)
        squash_line_segments(ax_D,     Ylimit_D,     K_D,  sign=-1,
                             color="grey", linewidth=0.8)


        ax_D.set_xlabel("fit_diff")
        ax_D.set_ylabel("D")
        ax_D.set_title("obs_diff vs fit_diff " + lVisible)
        fig_D.tight_layout()

        # ---------- Window 2 : normalised --------------------------------------------
        Dnorm_trans = squash(Y["Dnorm"], Ylimit_Dnorm, K_Dnorm)

        fig_Dnorm, ax_Dnorm = plt.subplots(figsize=(38.4, 21.6), dpi=100)
        ax_Dnorm.scatter(
            fit_norm,
            Dnorm_trans,
            s=6, marker=".", color=Y["color"], alpha=0.8
        )
        ax_Dnorm.axvline(0, color="grey", linewidth=0.8)
        ax_Dnorm.axhline(0, color="grey", linewidth=0.8)

        xmin, xmax = ax_Dnorm.get_xlim()
        x_vals = np.array([xmin, xmax])

        ax_Dnorm.set_xlim(xmin, xmax)
        ax_Dnorm.set_ylim(-Ylimit_Dnorm*1.3, Ylimit_Dnorm*1.3)   # ← pick your ymin,ymax here

        ax_Dnorm.axhline( Ylimit_Dnorm, color="grey", linewidth=0.8)
        ax_Dnorm.axhline(-Ylimit_Dnorm, color="grey", linewidth=0.8)

        squash_line_segments(ax_Dnorm, Ylimit_Dnorm, K_Dnorm, sign=+1,
                             color="grey", linewidth=0.8)
        squash_line_segments(ax_Dnorm, Ylimit_Dnorm, K_Dnorm, sign=-1,
                             color="grey", linewidth=0.8)

        ax_Dnorm.set_xlabel("fit_diff (norm)")
        ax_Dnorm.set_ylabel("D (norm)")
        ax_Dnorm.set_title("Normalised obs_diff vs fit_diff " + lVisible)
        fig_Dnorm.tight_layout()

        plt.show()

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(60, 20), constrained_layout=True
    )


    # ── left-hand histogram: raw D ────────────────────────────────────────────────
    ax1.hist(
        Y["D"], bins=601, alpha=0.8, color="steelblue", range=(-300, 300)
    )
    ax1.axvline(0, color="black", linewidth=0.8)          # central line at 0
    ax1.set_title("Histogram D = (c1−c2) − predicted " + lVisible)
    ax1.set_xlabel("D")
    ax1.set_ylabel("precincts")
    ax1.set_xlim(-200, 200)

    # ── right-hand histogram: normalised D ────────────────────────────────────────
    ax2.hist(
        Y["Dnorm"], bins=800, alpha=0.8, color="indianred", range=(-10, 10)
    )
    ax2.axvline(0, color="black", linewidth=0.8)          # central line at 0
    ax2.set_title("Histogram Dnorm = (c1−c2) − predicted (normalized) " + lVisible)
    ax2.set_xlabel("normalized D")
    ax2.set_ylabel("precincts")
    ax2.set_xlim(-10, 10)

    plt.show()

    nowCalc = datetime.now()

    # ------------------------------------------------------------
    # 8C.  List ±N outliers for D, relative D and probability of interval of confidence
    # ------------------------------------------------------------

    writer = pd.ExcelWriter(f"outliers{lVisible}{rok}.xlsx", engine="xlsxwriter")

    # ---------- helper: take any Series of scores ----------------
#    TOP_N = Y.shape[0]//3
    TOP_N = 500

    def sheet_name(label, sign):
        return f"{label}_{sign}"

    criterion = "Dnorm"

    large = Y.nlargest(TOP_N, criterion)
    small = Y.nsmallest(TOP_N, criterion)
    mid = Y.nsmallest(TOP_N*2, criterion).nlargest(TOP_N, criterion)

    def add_outliers(series: pd.Series, label: str,
                     sign : str,  k: int = TOP_N):
        slicer = series.nlargest(k) if "pos" == sign else series.nsmallest(k)
        #for sign, slicer in [("pos", series.nlargest(k)),
        #                     ("neg", series.nsmallest(k))]:
        sheet = sheet_name(label, sign)
        df = (
            slicer.rename("metric")
                  .to_frame()
                  .join(Y, how="left")     # keep all original cols
                  .reset_index()              # bring keys back as columns
        )
        df.to_excel(writer, sheet_name=sheet, index=False)

    # ---------- 6B.  ±N outliers for D, D_rel  ---------------
    add_outliers(Y["D"], "D", "pos")
    add_outliers(Y["D"], "D", "neg")
    add_outliers(Y["Dnorm"], "Dnorm", "pos")
    add_outliers(Y["Dnorm"], "Dnorm", "neg")
    top_rows = (
        Y[Y["D"] > 0]                  # 1. bierzemy tylko wiersze z D > 0
        .nlargest(TOP_N, "x_edge")     # 2. wybieramy TOP_N wg największego x_edge
    )
    top_rows.to_excel(writer, sheet_name="x_edge", index=False)
    #add_outliers(Y["x_edge"], "x_edge", "pos")

    # ---------- 6C.  save & finish -------------------------------
    writer.close()
    print(f"✓  All outlier tables written to outliers.xlsx")

    nowCalcEnd = datetime.now()

    print ('outliers time', nowCalcEnd-nowCalc)


    # CYFERKI

    if not cyferki:
        return

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
                if not pd.isna(row[e]) and ('all'==cyferki or row['class']==cyferki):
                    v = row[e]
                    if v < 50:
                        continue
                    s[round(v)%10] += 1
                    p[round(v)%5] += 1
                    count += 1
            if 300 <= count:
                s.append(count)
                p.append(count)
                histograms [nm + ' ' + titles[e]] = s
                pentagrams [nm + ' ' + titles[e]] = p


    #haveHistograms = [e for e in use2 if e in histograms]
    ccc = 0
    for key in histograms.keys():
        #if key not in pentagrams:                   # sanity check
        #    continue
        colour = palette[ccc % len(palette)]
        ccc += 1
        plot_histogram_pair(
            title=key,
            histo_data=histograms[key],
            penta_data=pentagrams[key],
            lVisible=lVisible + ' ' + cyferki,
            p_conf=0.95,
            bar_colour  = colour
        )

    #plot_histograms([(e, histograms[e]) for e in histograms], lVisible + ' ' + cyferki, p_conf=0.95
    #                )
    #plot_histograms([(e, pentagrams[e]) for e in pentagrams], lVisible + ' ' + cyferki, p_conf=0.95,
    #                category_labels=['0 i 5','1 i 6', '2 i 7', '3 i 8', '4 i 9'])

    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Demo: accept a '-c' flag plus positional arguments"
    )

    # Add the -c flag (no argument, just True/False)
    parser.add_argument(
        '-m',
        action='store_true',
        help="Histogramy główne"
    )
    parser.add_argument(
        '-H',
        action='store_true',
        help="Histogramy główne"
    )

    parser.add_argument(
        '-c',
        metavar='CYFERKI',
        help="Histogramy z cyferkami (specify argument here)"
    )
    parser.add_argument(
        '-y',
        metavar='CYFERKI',
        help="Histogramy z cyferkami (specify argument here)"
    )
    # Positional arguments (zero-or-more)
    parser.add_argument(
        'items',
        nargs='*',
        help='List of positional arguments'
    )

    args = parser.parse_args()

    print(f"-c flag present? {args.c}")
    global KEY1
    global KEY2
    rok = int(args.y)
    KEY1, KEY2 = terytGminy[rok], nrKomisji[rok]

    for idx, val in enumerate(args.items, 1):
        print(f"  {idx}: {val}")
        displaySomething (val, histogramy=args.H, cyferki=args.c, rok=rok,
                          mergedInfix=('-merged' if args.m else ''))
    
if __name__ == "__main__":
    main()
