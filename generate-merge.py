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
from pathlib import Path



fileGroups = [
    {'desc': 'w miastach ponad 250 tys. mieszkańców',
     'regrNm': 'YmiastoDst250+ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst250+ar2025.xlsx',
     kat:'250+'},
    {'desc': 'w miastach od 100 tys. do 250 tys. mieszkańców',
     'regrNm': 'YmiastoDst100-250+ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst100-250+ar2025.xlsx',
     kat:'100+'},
    {'desc': 'w miastach od 40 tys. do 100 tys. mieszkańców',
     'regrNm': 'YmiastoDst40-100ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst40-100ar2025.xlsx',
     kat:'40+'},
    {'desc': 'w miastach od 20 tys. do 40 tys. mieszkańców',
     'regrNm': 'YmiastoDst20-40ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst20-40ar2025.xlsx',
     kat:'20+'},
    {'desc': 'w miastach poniżej 20 tys. mieszkańców',
     'regrNm': 'YmiastoDst20-ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst20-ar2025.xlsx',
     kat:'20-'},
    {'desc': 'obwoty wiejskie i miejsco-wiejskie',
     'regrNm': 'YwiesDstar2025.xlsx',
     'outliersNm': 'outliers-YwiesDstar2025.xlsx',
     kat:'wieś'},
]

prokuraturyLista = pd.read_excel('prokuratury-lista.xlsx')

s = Path("subst")
s.mkdir(exist_ok=True)

def euro_fmt(x):
    """
    Format floats with:
       • thousands separator “.”
       • decimal separator “,”
       • two digits after the comma
    """
    return f"{x:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")


out = pd.DataFrame(index=prokuraturyLista.index)
out['nazwa'] = "Prokuratura Okręgowa " + prokuraturyLista["Prokuratura Okręgowa"]
out['dir'] = prokuraturyLista["Prokuratura Okręgowa"].str.replace(" ", "-", regex=False)
out['adres'] = out['nazwa']+r'\\'+prokuraturyLista["Adres"].str.replace(", ", r"\\", regex=False)

for g in fileGroups:
    g += pd.read_excel(
        g['outliersNm'],
        sheet_name=["x_edgePOS", "x_edgeNEG"])
    g['Coefficients'] = pd.read_excel(l['regrNm'], sheet_name='Coefficients')
    
#for l in fileGroups:
#    print (l)

THRESHOLDS =  [(0.99999, 'P'), (0.9999, 'X'), (0.999535, 'B'), (0.997300, 'N')]
EXPLAIN = {'P' : 'niemożliwym',
           'X' : 'ekstremalnie nieprawdopodobnym',
           'B': 'bardzo nieprawdopodobnym',
           'N' : 'nieprawdopodobnym'}
BENEF = {'POS' : "Karola NAWROCKIEGO", 'NEG' : "Rafała Trzaskowskiego"}

colsToKeep = ["Teryt Gminy", "Gmina", "Powiat", "Nr Komisji", "NAWROCKI Karol Tadeusz", "fitsNAWROCKI Karol Tadeusz",
              "TRZASKOWSKI Rafał Kazimierz", "fitsTRZASKOWSKI Rafał Kazimierz", "obs_diff", "fit_diff", "D", "diff_std", "x_edge"]

def euro_fmt(x):
    """
    Format floats with:
       • thousands separator “.”
       • decimal separator “,”
       • two digits after the comma
    """
    return f"{x:.6f}".replace(".", ",")

formatters = {c: euro_fmt for c in cols_to_keep if pd.api.types.is_float_dtype(df[c])}

for v in out:
    (s / v["dir"]).mkdir (parents=True, exist_ok=True)
    #extracts = {'POS': {l : {} for v, l in THRESHOLDS},
    #            'NEG': {l : {} for v, l in THRESHOLDS}}
    extracts = {'POS': {},'NEG': {}}
    for direction in ['POS', 'NEG']:
        probaMax = 2
        for probaMin, label in THRESHOLDS:
            for g in fileGroups:
                ou = g[x_edge+direction]
                ext = ou[(ou['x_adge'] >= probaMin) &(ou['x_adge'] < probaMax) & (ou['prokuratura']==v['nazwa'])]
                ext ['kat'] = g['kat']
                if not ext.empy:
                    if label in extracts[direction]:
                        extracts[direction][label] = pd.concat (
                            [extracts[direction][label], ext],
                            axis=0,
                            ignore_index=True)
                    else:
                        extracts[direction][label] = ext
            probaMax = probaMin
            if label in extracts[direction]:
                extracts[direction][label].to_latex (
                    columns=colsToKeep,
                    index=False,
                    longtable=True,
                    escape=True,
                    caption=f"Obwody z wynikiem  {EXPLAIN[label]} na korzyść {BENEF[direction]} według modelu" ,
                    label=f"tab:{direction}{label}",
                    
                    column_format = "llllllllllllllllll"
                    multicolumn=False
                    
                )
out.to_csv ('merge.csv', sep=';', index=True,
         header=True, encoding="utf-8",
         quoting=1, quotechar='"')
         

    
# Teryt Gminy; Gmina; Powiat; Nr komisji; Siedziba; Uprawnienie / Liczba wyborców uprawnionych do głosowania (umieszczonych w spisie, z uwzględnieniem dodatkowych formularzy) w chwili zakończenia głosowania;
# zaswiadczenia2t; w tym z powodu postawienia znaku „X” obok nazwisk obu kandydatów;
# NAWROCKI Karol Tadeusz; TRZASKOWSKI Rafał Kazimierz;
# fitsNAWROCKI Karol Tadeusz; fitsTRZASKOWSKI Rafał Kazimierz;
# obs_diff; fit_diff; D; diff_std; x_edge


# 3.0 -> 0.997300
# 3.5 -> 0.999535 ****
# 4.42 -> 0.99999 ****
# 3,89 -> 0.9999



#sheetGroups = [l for pd.read_excel("your_workbook.xlsx",
#                    sheet_name=["x_edgePOS", "x_edgeNEG"])
