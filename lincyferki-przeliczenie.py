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
import threading
import time
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

palette = plt.cm.tab10.colors

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

def classifyAlt (row):
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

def classify (row):
    countPis = 0
    a = row["member1_candidate"]
    if a in pis:
        countPis += 1
    if a in antypis:
        countPis -= 1

    if a != '' and not pd.isna (a) and a not in znaneKomitety:
        print ('Nieznany komitet', a)

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

use2 = [
    #'Liczba niewykorzystanych kart do głosowania',
    #'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym (liczba podpisów w spisie oraz adnotacje o\xa0wydaniu karty bez potwierdzenia podpisem w\xa0spisie)',
    #'Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym (łącznie)',
    #'Liczba wyborców głosujących na podstawie zaświadczenia o\xa0prawie do głosowania',
    #'Liczba kart wyjętych z\xa0urny',
    #'Liczba kart ważnych',
    #'Liczba głosów ważnych oddanych łącznie na obu kandydatów (z\xa0kart ważnych)',
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
    #ax.set_title(f"{title} | n={n} p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle} {lVisible}")
    ax.set_title(f"{title} | n={n} p={p_conf:.2f}, band=[{lo}, {hi}] {lVisible}")

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
    print ('h_n', h_n)
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
        #title=f"{title} 10-bins | n={h_n} {lVisible}",
        title=title,
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
        #title=f"{title} | n={p_n} p={p_conf:.2f}, band=[{lo}, {hi}]{subtitle} {lVisible}",
        title=title,
        p_conf=p_conf,
        bar_colour=bar_colour,
        lVisible=lVisible
    )

    fig.tight_layout()
    return fig            # return so caller can .savefig() if desired


def drawDrecount (Drecount):

    fig, ax1 = plt.subplots(
        nrows=1, ncols=1, figsize=(34, 20), constrained_layout=True
    )
    minVal = round(Drecount.min())
    maxVal = round(Drecount.max())
    print ('minVal', minVal, "maxVal", maxVal)
    ax1.set_xlim(minVal - 8, maxVal + 8)
    ax1.hist(
        Drecount,
        alpha=0.8, color="blue", bins=range(minVal-5, maxVal + 6))
    ax1.set_title("Różnica między kandydatami: rozbieżności między protokołami komisji a wynikami przeliczeń (dodatnie: błąd w protolole na korzyść Nawrockiego)")
    ax1.axvline(x=0, color='black', linewidth=1)
    plt.show(block=False)
    plt.pause(0.1)

def drawDrecountLowres (Drecount):

    fig, ax1 = plt.subplots(
        nrows=1, ncols=1, figsize=(17, 10), constrained_layout=True
    )
    minVal = round(Drecount.min())
    maxVal = round(Drecount.max())
    r = [v for v in range (minVal-5, maxVal+6) if v%4==0]
    ax1.set_xlim(minVal - 8, maxVal + 8)
    ax1.hist(
        Drecount,
        alpha=0.8, color="blue", bins=r)
    ax1.set_title("Różnica między kandydatami: rozbieżności między protokołami komisji a wynikami przeliczeń (dodatnie: błąd w protolole na korzyść Nawrockiego)")
    ax1.axvline(x=0, color='black', linewidth=1)
    plt.show(block=False)
    plt.pause(0.1)

def drawDratio (Dratio,
                ttl="Rozbieżność stwierdzona po przeliczeniu jako procent rozbieżności według naszego modelu",
                color="red"):

    fig, ax1 = plt.subplots(
        nrows=1, ncols=1, figsize=(17, 10), constrained_layout=True
    )
    minVal = (Dratio.min())
    maxVal = (Dratio.max())
    ax1.set_xlim(minVal - .03, maxVal + .03)
    ax1.hist(
        Dratio,
        alpha=0.8, color=color, range=(minVal, maxVal), bins=600)
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax1.set_title(ttl)
    ax1.axvline(x=0, color='black', linewidth=1)
    plt.show(block=False)
    plt.pause(0.1)

def drawTests (good, almostGood, bad, reverse, *, x_edgeMap=None, hiRes=True, haman=False,
               bar_frame=True, bw=False):
 
    dfs = [good, almostGood, bad, reverse]
    columnName = 'lp-haman total' if haman else 'lp-proba'
    try:
        columns = [good[columnName], almostGood[columnName], bad[columnName], reverse[columnName]]
    except:
        print ("GOOD COLUMNS", good.columns.to_list())
        raise
    labels = ['OK', 'prawie OK', 'błąd', 'błąd odwrotny']
    colors = ['#00dd88', '#99ff00', '#ff6644', '#dd00ff']
    #full_min = min(np.min(col) for col in columns)
    #full_max = max(np.max(col) for col in columns)

    full_min = min(
        np.nanmin( col.replace([np.inf, -np.inf], np.nan) )   # per-column min, NaNs skipped
        for col in columns
    )

    full_max = max(
        np.nanmax( col.replace([np.inf, -np.inf], np.nan) )   # per-column max, NaNs skipped
        for col in columns
    )


    
    if hiRes:
        font_size = 12
        tickCount = 25
        if full_max < 901:
            binCount = full_max+3
        else:
            binCount =800
    else:
        font_size = 15
        tickCount = 12
        if full_max < 281:
            binCount = full_max+3
        else:
            binCount = 250
    def pct_fmt(val: float) -> str:
        if full_max < 901 and hiRes:
            return f"{val * 100:,.3f}%".replace(".", ",")  # 0.11111 → 11,111 %
        elif full_max < 901 or hiRes:
            return f"{val * 100:,.3f}%".replace(".", ",")
        else:
            return f"{val * 100:,.2f}%".replace(".", ",")
    
    binWidth = math.ceil((full_max - full_min + 1) / (binCount-2))
    print ('dla próbek do', full_max, 'hiRes', hiRes, 'binWidth', binWidth, 'binCount', binCount)
    binLimits = []
    e = full_min-binWidth
    while e < full_max+1+binWidth:
        binLimits.append(e)
        e += binWidth
        #histograms = [np.histogram(col, bins=binLimits)[0] for col in columns]

    histograms = [
        np.histogram(col[np.isfinite(col)], bins=binLimits)[0]
        for col in columns
    ]
    counts_stacked = np.vstack(histograms)

    plt.rcParams.update({"font.size": font_size})

    figsizeY = (18 if hiRes else 10) if 5 < binWidth else (9 if hiRes else 5)
    fig, ax1 = plt.subplots(
        nrows=1, ncols=1, figsize=(31 if hiRes else 17, figsizeY), constrained_layout=True
    )
    bottom = np.zeros_like(histograms[0])
    if bw:
        # light-to-dark grey so the filled areas differ even on screen
        face_cols = ['0.7', '0.6', '0.4', '0.2']
        hatches   = ['', '//', 'xx', 'oo']
    else:
        face_cols = ['#00dd88', '#99ff00', '#ff4433', '#dd00ff']
        hatches   = [''] * 4
        
    for hist, color, hatch, label in zip(histograms, face_cols, hatches, labels):
        ax1.bar(
            binLimits[:-1],
            hist,
            width=binWidth,
            bottom=bottom,
            align='edge',
            color=color,
            hatch=hatch,
            edgecolor  = 'black' if bar_frame else face,
            linewidth  = 0.25 if bar_frame else 0.0,
            label=label)
        bottom += hist
    #ax1.set_xlim(full_min-binWidth, full_max+binWidth+1)
    ax1.set_xlim(full_min, full_max+1)
    ax1.xaxis.set_major_locator(MaxNLocator (nbins=13, integer=True))
    wedlug = "według J. Hamana" if haman else "(według naszego modelu)"
    ax1.set_xlabel("Obwody ponownie przeliczone, od najbardziej do najmniej prawdopodobnych nieprawidłowości " + wedlug)
    if 1 < binWidth:
        ax1.set_ylabel("Częstość")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Przeliczenia")
    ax1.legend()
    #ax1.grid(True, linestyle="--", linewidth=0.3)

    if not haman:
        ax_edge = ax1.twiny()
        ax_edge.xaxis.set_ticks_position("bottom")
        ax_edge.xaxis.set_label_position("bottom")
        ax_edge.spines["bottom"].set_position(("outward", 40))  # 25 pt lower
        ax_edge.set_xlim(ax1.get_xlim())

        tick_idx   = np.linspace(0, len(binLimits) - 1, tickCount, dtype=int)
        lp_ticks   = [int(round(binLimits[i])) for i in tick_idx]
        ax_edge.set_xticks(lp_ticks)
        mapping = (
            x_edgeMap[["lp-proba", "x_edge"]]
            .set_index("lp-proba")
            .sort_index()
        )

        edge_labels = [
            pct_fmt(mapping.loc[lp, "x_edge"]) if lp in mapping.index else (pct_fmt(1.0) if lp < 20 else "")
            for lp in lp_ticks
        ]

        ax_edge.set_xticklabels(edge_labels)
        ax_edge.set_xlabel("Prawdopodobieństwo nieprawidłowości")

    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    plt.pause(0.1)
    plt.pause(0.1)

def displaySomething ():
    #Y = pd.read_excel(DATA_DIR / f"Y{l}C{rok}{mergedInfix}.xlsx")
    Y = pd.read_excel(DATA_DIR / "subst" / "nieprawdopodobne2.xlsx")
    Y["lp-proba"] = Y.reset_index().index+1
    przeliczenia = pd.read_excel(DATA_DIR / "Tabela_250_okw.xlsx")

    joined = Y.merge(
        przeliczenia,
        how="left",
        left_on=["Siedziba", "Nr komisji"],
        right_on=["Siedziba komisji", "Numer komisji"],
        indicator=True,
        suffixes=("", "_prokurator")
    )
    merge_check = przeliczenia.merge(
        Y,
        how="left",
        left_on=["Siedziba komisji", "Numer komisji"],
        right_on=["Siedziba", "Nr komisji"]
    )
    counts = merge_check.groupby(merge_check.columns[:2].tolist()).size().reset_index(name='match_count')
    warn_rows = counts[counts["match_count"] != 1]
    print("Warning: The following rows in `przeliczenia` matched 0 or more than 1 row in `Y`:")
    print(warn_rows)

    # Haman

    outExcel = pd.ExcelWriter('nieprawdopodobne2przel.xlsx', engine="xlsxwriter")
    
    hamanLeftCols  = ["Gmina",
                      "Nr komisji",
                      "Liczba kart ważnych",
                      "NAWROCKI Karol Tadeusz",
                      "TRZASKOWSKI Rafał Kazimierz"]
    
    hamanRightCols = ["gmina",
                      "nr",
                      "karty wazne",
                      "Naw",
                      "Trza"]
    
    
    haman = pd.read_excel(DATA_DIR / "haman-converte-fixed.xlsx")
    haman["lp-haman total"] = haman["lp-haman"] + (haman["tabl#"]-1)*145
    renameDict = {c: f"haman-{c}" for c in haman.columns if not 'haman' in c}
    haman = haman.rename(columns=renameDict)
    hamanRightColsRenamed = [renameDict.get(c, c) for c in hamanRightCols]

    joined = (
        joined.merge(
            haman,
            how="left",
            left_on=hamanLeftCols,
            right_on=hamanRightColsRenamed,
            validate="one_to_one",          # ensures no duplicates on either side
            indicator="_mergeH"
         )
    )

    joined.to_excel (outExcel, sheet_name='joinedDebug', index=True)
    
    # --------------------------------------------------------------------------
    # 3.  rows of *haman* that never matched  →  rejects
    # --------------------------------------------------------------------------
    rejects = (
        haman.merge(
            joined[hamanLeftCols],
            how="left",
            left_on=hamanRightColsRenamed,
            right_on=hamanLeftCols,
            indicator="_mergeH2"
        )
        .query('_mergeH2 == "left_only"')
    )

    recounted = joined[joined["_merge"] == "both"]

    Drecount = recounted["Drecount"] = recounted["Rafał TrzaskowskiD"] - recounted["Karol NawrockiD"]
    recounted["DrecountPlus"] = recounted["Drecount"].apply (lambda x : x if x >= 0 else 0)
    recounted["DrecountMinus"] = recounted["Drecount"].apply (lambda x : x if 0 >= x else 0)
    Dratio = recounted["Dratio"] = recounted["Drecount"] / recounted["D"]
    
    DratioHaman = recounted ["DratioHaman"] = recounted["Drecount"] / recounted ["haman-blad"]
    
    cols = ["x_edge", "lp-proba", "lp-haman total"]
    colsProba = ["x_edge", "lp-proba"]
    good = recounted[recounted["NIE"]==1][cols]
    almostGood = recounted[(recounted["TAK"]==1)
                           & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                              .apply (lambda n: -2 <= n and n <= 2))][cols]
    bad = recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["Dratio"] > 0)][cols]
    badHaman = recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["DratioHaman"] > 0)][cols]
    reverse =  recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["Dratio"] < 0)][cols]
    reverseHaman =  recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["DratioHaman"] < 0)][cols]
    #drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[colsProba], hiRes=False)
    drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True)
    drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True, bw=True)
    #drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[colsProba], hiRes=True)

    #drawTests (good, almostGood, badHaman, reverseHaman, haman=True, hiRes=False)
    drawTests (good, almostGood, badHaman, reverseHaman, haman=True, hiRes=False, bar_frame=True)
    drawTests (good, almostGood, badHaman, reverseHaman, haman=True, hiRes=False, bar_frame=True, bw=True)
    #drawTests (good, almostGood, badHaman, reverseHaman, haman=True, hiRes=True)
    zoom = 3300
    goodX = good[good['lp-proba'] < zoom]
    almostGoodX = almostGood[almostGood['lp-proba'] < zoom]
    badX = bad[bad['lp-proba'] < zoom]
    reverseX = reverse[reverse['lp-proba'] < zoom]
    #drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[colsProba], hiRes=False)
    drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True)
    drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True, bw=True)
    #drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[colsProba], hiRes=True)
    zoom = 240
    goodX2 = good[good['lp-proba'] < zoom]
    almostGoodX2 = almostGood[almostGood['lp-proba'] < zoom]
    badX2 = bad[bad['lp-proba'] < zoom]
    reverseX2 = reverse[reverse['lp-proba'] < zoom]
    #drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[colsProba], hiRes=False)
    drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True)
    drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[colsProba], hiRes=False, bar_frame=True, bw=True)
    #drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[colsProba], hiRes=True)
    frauds = recounted[recounted["TAK"]==1]
    #drawDrecount(Drecount)
    #drawDrecountLowres(Drecount)
    #drawDratio(Dratio)

    fraudsImpossible = frauds [frauds['x_edge'] >= 0.99999]
    print ('frauds', frauds.shape[1], 'fraudsImpossible', fraudsImpossible.shape[1])
    #drawDratio(fraudsImpossible['Dratio'],
    #           "Rozbieżność stwierdzona po przeliczeniu dla wynikó°w niemożliwych jako procent rozbieżności według naszego modelu",
    #           "#aa8800")
    


    
        
    recounted.to_excel (outExcel, sheet_name='recounted', index=True)

    frauds.to_excel (outExcel, sheet_name='frauds', index=True)
    fraudsImpossible.to_excel (outExcel, sheet_name='fraudsImpossible', index=True)


    
    joined.to_excel (outExcel, sheet_name='Y', index=True)
    rejects.to_excel (outExcel, sheet_name='hamanRejects', index=True)

    joinedHaman = joined.sort_values(by=["haman-tabl#", "lp-haman"],
                                     na_position="last",
                                     kind="mergesort")
    
    joinedHaman.to_excel (outExcel, sheet_name='Yhaman', index=True)
    outExcel.close()
    
    return

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
        '-w',
        action='store_true',
        help="województwa"
    )

    parser.add_argument(
        '-W',
        action='store_true',
        help="Warszawa"
    )

    parser.add_argument(
        '-D',
        action='store_true',
        help="sign of difference D = reported - rsult of lin regression"
    )

    parser.add_argument(
        '-d',
        action='store_true',
        help="who won?"
    )

    parser.add_argument(
        '-o',
        action='store_true',
        help="output outliers"
    )

    parser.add_argument(
        '-l',
        type=int,
        default=0,
        metavar='LUDNOSC',
        help="Warszawa"
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
    if args.y:
        rok = int(args.y)
    else:
        rok = 2025
    KEY1, KEY2 = terytGminy[rok], nrKomisji[rok]
    displaySomething()
    input ("introduisez votre sexe dans la machine")
        
if __name__ == "__main__":
    main()

# kujawsko-pomorskie lubelskie mazowieckie warm-maz wielkop
