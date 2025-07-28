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

def drawTests (good, almostGood, bad, reverse, *, x_edgeMap, hiRes=True):
 
    dfs = [good, almostGood, bad, reverse]
    columns = [good['lp-proba'], almostGood['lp-proba'], bad['lp-proba'], reverse['lp-proba']]
    labels = ['OK', 'prawie OK', 'błąd', 'błąd odwrotny']
    colors = ['#00dd88', '#99ff00', '#ff6644', '#dd00ff']
    full_min = min(np.min(col) for col in columns)
    full_max = max(np.max(col) for col in columns)

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
    binLimits = []
    e = full_min-binWidth
    while e < full_max+1+binWidth:
        binLimits.append(e)
        e += binWidth
    histograms = [np.histogram(col, bins=binLimits)[0] for col in columns]
    counts_stacked = np.vstack(histograms)

    plt.rcParams.update({"font.size": font_size})
    
    fig, ax1 = plt.subplots(
        nrows=1, ncols=1, figsize=(34 if hiRes else 17, 20 if hiRes else 10), constrained_layout=True
    )
    bottom = np.zeros_like(histograms[0])
    for hist, color, label in zip(histograms, colors, labels):
        ax1.bar(binLimits[:-1], hist, width=binWidth, bottom=bottom, color=color, label=label, align='edge')
        bottom += hist
    ax1.set_xlim(full_min-binWidth, full_max+binWidth+1)
    ax1.set_xlabel("Obwody ponownie przeliczone, od najbardziej do najmniej prawdopodobnych nieprawidłowości (według naszego modelu)")
    ax1.set_ylabel("Częstość")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Przeliczenia")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.3)

    ax_edge = ax1.twiny()
    ax_edge.xaxis.set_ticks_position("bottom")
    ax_edge.xaxis.set_label_position("bottom")
    ax_edge.spines["bottom"].set_position(("outward", 40))  # 25 pt lower
    ax_edge.set_xlim(ax1.get_xlim())

    mapping = (
        x_edgeMap[["lp-proba", "x_edge"]]
        .set_index("lp-proba")
        .sort_index()
    )

    tick_idx   = np.linspace(0, len(binLimits) - 1, tickCount, dtype=int)
    lp_ticks   = [int(round(binLimits[i])) for i in tick_idx]
    edge_labels = [
        pct_fmt(mapping.loc[lp, "x_edge"]) if lp in mapping.index else (pct_fmt(1.0) if lp < 20 else "")
        for lp in lp_ticks
    ]

    ax_edge.set_xticks(lp_ticks)
    ax_edge.set_xticklabels(edge_labels)
    ax_edge.set_xlabel("Prawdopodobieństwo nieprawidłowości")
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def displaySomething ():
    #Y = pd.read_excel(DATA_DIR / f"Y{l}C{rok}{mergedInfix}.xlsx")
    Y = pd.read_excel(DATA_DIR / "subst" / "nieprawdopodobne2.xlsx")
    Y["lp-proba"] = Y.reset_index().index+1
    przeliczenia = pd.read_excel(DATA_DIR / "Tabela_250_okw.xlsx")

    recounted = Y.merge(
        przeliczenia,
        how="inner",
        left_on=["Siedziba", "Nr komisji"],
        right_on=["Siedziba komisji", "Numer komisji"],
        indicator=True,
        suffixes=("", "_prokurator")
    )
    Drecount = recounted["Drecount"] = recounted["Rafał TrzaskowskiD"] - recounted["Karol NawrockiD"]
    recounted["DrecountPlus"] = recounted["Drecount"].apply (lambda x : x if x >= 0 else 0)
    recounted["DrecountMinus"] = recounted["Drecount"].apply (lambda x : x if 0 >= x else 0)
    Dratio = recounted["Dratio"] = recounted["Drecount"] / recounted["D"]
    cols = ["x_edge", "lp-proba"]
    good = recounted[recounted["NIE"]==1][cols]
    almostGood = recounted[(recounted["TAK"]==1)
                           & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                              .apply (lambda n: -2 <= n and n <= 2))][cols]
    bad = recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["Dratio"] > 0)][cols]
    reverse =  recounted[(recounted["TAK"]==1)
                    & ((recounted["Karol NawrockiD"]-recounted["Rafał TrzaskowskiD"])
                       .apply (lambda n: n < -2  or 2 < n))
                    & (recounted["Dratio"] < 0)][cols]
    drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[cols], hiRes=False)
    drawTests (good, almostGood, bad, reverse, x_edgeMap=Y[cols], hiRes=True)
    zoom = 3300
    goodX = good[good['lp-proba'] < zoom]
    almostGoodX = almostGood[almostGood['lp-proba'] < zoom]
    badX = bad[bad['lp-proba'] < zoom]
    reverseX = reverse[reverse['lp-proba'] < zoom]
    drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[cols], hiRes=False)
    drawTests (goodX, almostGoodX, badX, reverseX, x_edgeMap=Y[cols], hiRes=True)
    zoom = 240
    goodX2 = good[good['lp-proba'] < zoom]
    almostGoodX2 = almostGood[almostGood['lp-proba'] < zoom]
    badX2 = bad[bad['lp-proba'] < zoom]
    reverseX2 = reverse[reverse['lp-proba'] < zoom]
    drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[cols], hiRes=False)
    drawTests (goodX2, almostGoodX2, badX2, reverseX2, x_edgeMap=Y[cols], hiRes=True)
    frauds = recounted[recounted["TAK"]==1]
    #drawDrecount(Drecount)
    #drawDrecountLowres(Drecount)
    #drawDratio(Dratio)

    fraudsImpossible = frauds [frauds['x_edge'] >= 0.99999]
    print ('frauds', frauds.shape[1], 'fraudsImpossible', fraudsImpossible.shape[1])
    #drawDratio(fraudsImpossible['Dratio'],
    #           "Rozbieżność stwierdzona po przeliczeniu dla wynikó°w niemożliwych jako procent rozbieżności według naszego modelu",
    #           "#aa8800")
    
    
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

    outExcel = pd.ExcelWriter('nieprawdopodobne2przel.xlsx', engine="xlsxwriter")
    
    joined.to_excel (outExcel, sheet_name='Y', index=True)
    
    recounted.to_excel (outExcel, sheet_name='recounted', index=True)

    frauds.to_excel (outExcel, sheet_name='frauds', index=True)
    fraudsImpossible.to_excel (outExcel, sheet_name='fraudsImpossible', index=True)
    outExcel.close()

    return

    
    if mergedInfix:
        print ('classify')
        Y["class"] = Y.apply(classify, axis=1)
    else:
        print ('NO classify')
        Y["class"] = "black";
        #Y.loc[:, "class"] = "black"
    Y["color"] = Y["class"].map(colors)

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

        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, figsize=(60, 20), constrained_layout=True
        )

        # ── left-hand histogram: raw D ────────────────────────────────────────────────
        #vals = ((Y["D"] + 2) // 4).astype(int)       # the integer data
        edges = np.arange(-302.5, 301.5, 6)
        ax1.hist(
            Y["D"], bins=edges, alpha=0.8, color="steelblue")
        ax1.set_xlim(-100, 100)
        ax1.axvline(0, color="black", linewidth=0.8)          # central line at 0
        ax1.set_title("Histogram D = (c1−c2) − predicted (grouped)" + lVisible)
        ax1.set_xlabel("D")
        ax1.set_ylabel("precincts")
        ax1.set_xlim(-300, 300)

        plt.show()
        
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(60, 20), constrained_layout=True
        )


        # ── left-hand histogram: raw D ────────────────────────────────────────────────
        ax1.hist(
            Y["Dnaw"],
            bins=601, alpha=0.8, color="blue", range=(-300, 300)
        )
        ax1.axvline(0, color="black", linewidth=0.8)          # central line at 0
        ax1.set_title("Histogram delta NAW = NAWROCKI − predicted " + lVisible)
        ax1.set_xlabel("D NAW")
        ax1.set_ylabel("precincts")
        ax1.set_xlim(-200, 200)

        # ── right-hand histogram: normalised D ────────────────────────────────────────
        ax2.hist(
            Y["Dnaw_norm"], bins=800, alpha=0.8, color="steelblue", range=(-10, 10)
        )
        ax2.axvline(0, color="black", linewidth=0.8)          # central line at 0
        ax2.set_title("Histogram delta NAW norm = NAWROCKI − predicted (normalized) " + lVisible)
        ax2.set_xlabel("normalized D NAW")
        ax2.set_ylabel("precincts")
        ax2.set_xlim(-10, 10)

        plt.show()


        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(60, 20), constrained_layout=True
        )


        # ── left-hand histogram: raw D ────────────────────────────────────────────────
        ax1.hist(
            Y["Dtrza"],
            bins=601, alpha=0.8, color="red", range=(-300, 300)
        )
        ax1.axvline(0, color="black", linewidth=0.8)          # central line at 0
        ax1.set_title("Histogram delta TRZA = TRZASKOWSKI − predicted " + lVisible)
        ax1.set_xlabel("D TRZA")
        ax1.set_ylabel("precincts")
        ax1.set_xlim(-200, 200)

        # ── right-hand histogram: normalised D ────────────────────────────────────────
        ax2.hist(
            Y["Dtrza_norm"], bins=800, alpha=0.8, color="indianred", range=(-10, 10)
        )
        ax2.axvline(0, color="black", linewidth=0.8)          # central line at 0
        ax2.set_title("Histogram delta TRZA norm = TRZASKOWSKI − predicted (normalized) " + lVisible)
        ax2.set_xlabel("normalized D TRZA")
        ax2.set_ylabel("precincts")
        ax2.set_xlim(-10, 10)

        plt.show()


    nowCalc = datetime.now()

    # ------------------------------------------------------------
    # 8C.  List ±N outliers for D and for relative D
    # ------------------------------------------------------------

    if addOutliers:
        writer = pd.ExcelWriter(f"outliers-{filename}", engine="xlsxwriter")

        # ---------- helper: take any Series of scores ----------------
        #    TOP_N = Y.shape[0]//3
        TOP_N = 400

        def sheet_name(label, sign):
            return f"{label}_{sign}"

        criterion = "Dnorm"

        large = Y.nlargest(TOP_N, criterion)
        small = Y.nsmallest(TOP_N, criterion)
        #mid = Y.nsmallest(TOP_N*2, criterion).nlargest(TOP_N, criterion)

        def add_outliers(series: pd.Series, label: str, sign : str,  k: int = TOP_N):
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

        # ---------- 6B.  ±100 outliers for D and D_rel ---------------
        add_outliers(Y["D"], "D", "pos")
        add_outliers(Y["D"], "D", "neg")
        add_outliers(Y["Dnorm"], "Dnorm", "pos")
        add_outliers(Y["Dnorm"], "Dnorm", "neg")
        #add_outliers(Y["D"],      "D")
        #add_outliers(Y["Dnorm"],  "Dnorm")
        top_rows = (
            Y[Y["D"] > 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "x_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        top_rows.to_excel(writer, sheet_name="x_edgePOS", index=False)
        bottom_rows = (
            Y[Y["D"] < 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "x_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        bottom_rows.to_excel(writer, sheet_name="x_edgeNEG", index=False)

        
        add_outliers(Y["Dnaw"], "Dnaw", "pos")
        add_outliers(Y["Dnaw"], "Dnaw", "neg")
        add_outliers(Y["Dnaw_norm"], "Dnaw_norm", "pos")
        add_outliers(Y["Dnaw_norm"], "Dnaw_norm", "neg")
        top_rows = (
            Y[Y["Dnaw"] > 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "naw_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        top_rows.to_excel(writer, sheet_name="naw_edgePOS", index=False)
        bottom_rows = (
            Y[Y["Dnaw"] < 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "naw_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        bottom_rows.to_excel(writer, sheet_name="naw_edgeNEG", index=False)

        
        add_outliers(Y["Dtrza"], "Dtrza", "pos")
        add_outliers(Y["Dtrza"], "Dtrza", "neg")
        add_outliers(Y["Dtrza_norm"], "Dtrza_norm", "pos")
        add_outliers(Y["Dtrza_norm"], "Dtrza_norm", "neg")
        top_rows = (
            Y[Y["Dtrza"] > 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "trza_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        top_rows.to_excel(writer, sheet_name="trza_edgePOS", index=False)
        bottom_rows = (
            Y[Y["Dtrza"] < 0]                  # 1. bierzemy tylko wiersze z D > 0
            .nlargest(TOP_N, "trza_edge")     # 2. wybieramy TOP_N wg największego x_edge
        )
        bottom_rows.to_excel(writer, sheet_name="trza_edgeNEG", index=False)

        
        # ---------- 6C.  save & finish -------------------------------
        writer.close()
        print(f"✓  All outlier tables written to outliers.xlsx")

    nowCalcEnd = datetime.now()

    print ('outliers time', nowCalcEnd-nowCalc)


    # CYFERKI

    if not cyferki:
        return
    printed = set()
    ludnoscAboveLabel = f">= {ludnosc}"
    ludnoscBelowLabel = f"< {ludnosc}"
    
    histograms = {}
    pentagrams = {}
    for ttt, nm in [(Y, 'all')]:
        for e in use2:
            s = {}
            p = {}
            count = {}
            for idx, row in ttt.iterrows():
                smallKey = row['Województwo'] if wojewodztwa else ''
                if warszawa and 'Warszawa' == row['Powiat']:
                    smallKey = 'Warszawa'
                #print ('smallkey', smallKey, type (smallKey), 's', s, type(s))
                if ludnosc:
                    gm = row['Gmina']
                    if gm not in printed:
                        printed.add(gm)
                        if row["Ludnosc"] < ludnosc:
                            None
                            #print ('small', row['Gmina'], row["Ludnosc"], ludnosc)
                        else:
                            print  ('BIG', row['Gmina'])
                    smallKey += ludnoscBelowLabel if row["Ludnosc"] < ludnosc else ludnoscAboveLabel
                if diff:
                    smallKey += "NAW wygrywa" if row['TRZASKOWSKI Rafał Kazimierz'] < row ['NAWROCKI Karol Tadeusz'] else "TRZA wygrywa"
                if diffRegr:
                    smallKey += "big D" if 0.2<row['Dnorm'] else "smallD "
                if smallKey not in s:
                    s[smallKey] = [0]*10
                    p[smallKey] = [0]*5
                    count[smallKey] = 0
                    
                if not pd.isna(row[e]) and ('all'==cyferki or row['class']==cyferki):
                    v = row[e]
                    if v < 50:
                        continue
                    s[smallKey][round(v)%10] += 1
                    p[smallKey][round(v)%5] += 1
                    count[smallKey] += 1
            for k in s:
                if 200 <= count[k]:
                    s[k].append(count[k])
                    p[k].append(count[k])
                    if k not in histograms:
                        histograms[k] = {}
                        pentagrams[k] = {}
                    histograms [k][nm + ' ' + titles[e]] = s[k]
                    pentagrams [k][nm + ' ' + titles[e]] = p[k]


    #haveHistograms = [e for e in use2 if e in histograms]
    for k in histograms:
        ccc = 0
        for key in histograms[k].keys():
            #if key not in pentagrams:                   # sanity check
            #    continue
            colour = palette[ccc % len(palette)]
            ccc += 1
            plot_histogram_pair(
                title=k+' '+key,
                histo_data=histograms[k][key],
                penta_data=pentagrams[k][key],
                lVisible=lVisible + ' ' + cyferki,
                p_conf=0.95,
                bar_colour  = colour
            )
        plt.show()

    #plot_histograms([(e, histograms[e]) for e in histograms], lVisible + ' ' + cyferki, p_conf=0.95
    #                )
    #plot_histograms([(e, pentagrams[e]) for e in pentagrams], lVisible + ' ' + cyferki, p_conf=0.95,
    #                category_labels=['0 i 5','1 i 6', '2 i 7', '3 i 8', '4 i 9'])


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
    input ("introduises votre sexe dans la machine")
        
if __name__ == "__main__":
    main()

# kujawsko-pomorskie lubelskie mazowieckie warm-maz wielkop
