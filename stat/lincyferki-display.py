import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import mahalanobis
from scipy.signal import find_peaks
import math
import statistics
from datetime import datetime
import argparse

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

def displaySomething (l, *, histogramy, cyferki, rok, mergedInfix):
    lVisible = l or 'wszystko'
    Y = pd.read_excel(DATA_DIR / f"Y{l}B{rok}{mergedInfix}.xlsx")
    
    nowAfterInit = datetime.now()
    #print (nowAfterInit-nowStart)

    nowRead = datetime.now()

    #print ('reading time', nowRead-nowStart)
    denom = (Y[c[rok][0]] + Y[c[rok][1]]) ** 0.5
    obs_norm  = Y["obs_diff"]  / denom
    fit_norm  = Y["fit_diff"]  / denom

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
        help="Z podziałem na województwa"
    )
    parser.add_argument(
        '-W',
        action='store_true',
        help="z oddizeleniem Warszawy"
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
                          wojewodztwa=args.w, warszawa=args.W,
                          mergedInfix=('-merged' if args.m else ''))
    
if __name__ == "__main__":
    main()
