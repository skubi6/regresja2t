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

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
#from pandas.io.formats.latex_formatter import escape_latex
#try:                                    # pandas 1.3 … 2.3.1 wheel
#    from pandas.io.formats.latex import escape_latex
#except ModuleNotFoundError:
#    try:                                # niektóre buildy 2.1 – 2.2 dfsg
#        from pandas.io.formats.format import escape_latex
#    except ModuleNotFoundError:         # bardzo nowe dev-gałęzie
#        from pandas.io.formats.latex_formatter import escape_latex

import re

_LATEX_SPECIALS = re.compile(r'([#$%&~_^\\{}])')

def escape_latex(text) -> str:
    return _LATEX_SPECIALS.sub(r'\\\1', str(text))

def escape_latexNomath(text) -> str:
    if '$' in text:
        return text
    else:
        return _LATEX_SPECIALS.sub(r'\\\1', str(text))



fileGroups = [
    {'desc': 'w miastach ponad 250 tys. mieszkańców',
     'regrNm': 'YmiastoDst250+ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst250+ar2025.xlsx',
     'kat':'250+', 'filesuffix': '250P'},
    {'desc': 'w miastach od 100 tys. do 250 tys. mieszkańców',
     'regrNm': 'YmiastoDst100-250+ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst100-250+ar2025.xlsx',
     'kat':'100+', 'filesuffix': '100P'},
    {'desc': 'w miastach od 40 tys. do 100 tys. mieszkańców',
     'regrNm': 'YmiastoDst40-100ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst40-100ar2025.xlsx',
     'kat':'40+', 'filesuffix': '40P'},
    {'desc': 'w miastach od 20 tys. do 40 tys. mieszkańców',
     'regrNm': 'YmiastoDst20-40ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst20-40ar2025.xlsx',
     'kat':'20+', 'filesuffix': '20P'},
    {'desc': 'w miastach poniżej 20 tys. mieszkańców',
     'regrNm': 'YmiastoDst20-ar2025.xlsx',
     'outliersNm': 'outliers-YmiastoDst20-ar2025.xlsx',
     'kat':'20-', 'filesuffix': '20M'},
    {'desc': 'obwoty wiejskie i miejsco-wiejskie',
     'regrNm': 'YwiesDstar2025.xlsx',
     'outliersNm': 'outliers-YwiesDstar2025.xlsx',
     'kat':'wieś', 'filesuffix': 'wies'},
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

def latinize_pl(text: str) -> str:
    PL2LAT = str.maketrans(
        "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
        "acelnoszzACELNOSZZ"
    )
    return text.translate(PL2LAT)

renameRegrAux = {
    "Intercept" : "wyraz wolny",
    "uwzględnieniem dodatkowych formularzy" : "Uprawnieni (w tym zaświadczenia 1t)",
    "obok nazwiska dwóch lub większej liczby kandydatów": "dwa znaki X",
    "Liczba wyborców głosujących na podstawie zaświadczenia": "Zaświadczenia 1t",
    "Liczba głosów nieważnych" : "Liczba głosów nieważnych",
}
    
    

def renameRegr (s):
    for k in renameRegrAux:
        #print ('test', '<'+k+'>', '<'+s+'>')
        if k in s:
            #print ('TAK')
            return renameRegrAux[k]
    return s

out = pd.DataFrame(index=prokuraturyLista.index)
out['nazwa'] = "Prokuratura Okręgowa " + prokuraturyLista["Prokuratura Okręgowa"]
out['dir'] = prokuraturyLista["Prokuratura Okręgowa"].str.replace(" ", "-", regex=False).apply(latinize_pl)
out['adres'] = out['nazwa']+r'\\'+prokuraturyLista["Adres"].str.replace(", ", r"\\", regex=False)


globalVals = {'PgAlt':0, 'Pg': 0, 'XgAlt':0, 'Xg': 0, 'BgAlt':0, 'Bg': 0,
              'NgAlt':0, 'Ng': 0, 'sumg' : 0, 'sumgAlt': 0,
              'Pincr': 0, 'PincrAlt': 0, 'Xincr': 0,'XincrAlt': 0,
              'Bincr': 0, 'BincrAlt': 0, 'Nincr': 0, 'NincrAlt': 0,
              'Gg': 0}
    

#FONT_NAME = "LMRoman10"
#FONT_PATH = Path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf")
#FONT_PATH = Path("/usr/share/texmf/fonts/truetype/public/lm/lmroman10-regular.ttf")

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"		
FONT_NAME = "DejaVuSerif"
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

PT_PER_CM = 72.27 / 2.54          # 28.452  pt ≈ 1 cm in TeX world
LIMIT_CM  = 2.4                   # ← change: physical width you allow
LIMIT_PT  = LIMIT_CM * PT_PER_CM  # threshold in points

def text_width_pt(text: str, size_pt: float = 10) -> float:
    return pdfmetrics.stringWidth(text, FONT_NAME, size_pt)

def split_middle(val: str) -> str:
    """
    Return a LaTeX-safe string.
    If it contains 'gm. …' or 'm. …' and is long, insert one `\\`
    at the blank or hyphen closest to the midpoint.
    """
    if not isinstance(val, str):
        return val                       # leave NaN etc. unchanged

    # Candidate split positions (space or dash, but NOT the first one after gm./m.)
    cand = [m.start() for m in re.finditer(r'[ \-]', val)]
    if text_width_pt(val) <= LIMIT_PT:
        return escape_latex(val)
    if not cand:
        print ('no cand')        
        return escape_latex(val)
    # Choose the one nearest the midpoint
    mid   = len(val) // 2
    pos   = min(cand, key=lambda i: abs(i - mid))
    char  = val[pos]

    first  = escape_latex(val[:pos])
    second = escape_latex(val[pos + 1 :])

    if char == " ":
        inner = rf"{first}\\{second}"     # blank gets replaced
    else:  # char == '-'
        inner = rf"{first}-\\{second}"  # keep '-' then break
    return fr"\makecell[l]{{{inner}}}"

def euro_fmt(x):
    return f"{x:.6f}".replace(".", ",")

def euro_fmt1(x):
    return f"{x:.1f}".replace(".", ",")

pl_msg   = "ciąg dalszy na następnej stronie"

coefFormatters = {
    'Nawrocki': euro_fmt,
    'Trzaskowski': euro_fmt,
    'nieważne': euro_fmt,
    "dwa X": euro_fmt,
}
for g in fileGroups:
    g |= pd.read_excel(
        g['outliersNm'],
        sheet_name=["x_edgePOS", "x_edgeNEG"])
    regr = pd.read_excel(g['regrNm'], sheet_name='Coefficients')
    g['Coefficients'] = regr
    regrPrint = regr.copy()
    regrPrint["Unnamed: 0"] = regrPrint["Unnamed: 0"].map(renameRegr)
    regrPrint.rename(columns={"Unnamed: 0" : " ",
                              "NAWROCKI" : "Nawrocki",
                              "TRZASKOWSKI" : "Trzaskowski"}, inplace=True)
    regrTxt = regrPrint.to_latex (
        index=False,
        longtable=True,
        escape=True,
        caption=rf"Tablica współczynników regresji, obwody {{\textbf {{{g['kat']}}}}} ({g['desc']})",
        label=f"tab:coef{g['filesuffix']}",
        column_format = "lrrrrrrrrr",
        multicolumn=False
    )
    regrTxt = regrTxt.replace("Continued on next page", pl_msg)
    with open(s / f"coeff{g['filesuffix']}.tex", "w", encoding="utf-8") as f:
        f.write(regrTxt)
    
    globalVals['Gg'] += pd.read_excel(g['regrNm'], sheet_name='Y').shape[0]
    
#for l in fileGroups:
#    print (l)

THRESHOLDS =  [(0.99999, 'P'), (0.9999, 'X'), (0.999535, 'B'), (0.997300, 'N')]
EXPLAIN = {'P' : 'niemożliwym',
           'X' : 'ekstremalnie nieprawdopodobnym',
           'B': 'bardzo nieprawdopodobnym',
           'N' : 'nieprawdopodobnym'}
BENEF = {'POS' : "Karola Nawrockiego", 'NEG' : "Rafała Trzaskowskiego"}

colsToKeepBase = ["kat", "Teryt Gminy", "Gmina", "Powiat",
                  "Nr komisji", "zaswiadczenia2t", "NAWROCKI Karol Tadeusz",
                  "fitsNAWROCKI Karol Tadeusz",
                  "TRZASKOWSKI Rafał Kazimierz",
                  "fitsTRZASKOWSKI Rafał Kazimierz",
                  "obs_diff", "fit_diff", "D",
                  "diff_std", "x_edge"]

recols = {
    "NAWROCKI Karol Tadeusz" : "NAW",
    "TRZASKOWSKI Rafał Kazimierz" : "TRZA",
    "fitsNAWROCKI Karol Tadeusz" : "fitNAW",
    "fitsTRZASKOWSKI Rafał Kazimierz" : "fitTRZA",
    "Teryt Gminy" : "Teryt",
    "Nr komisji" : "Nr",
    "obs_diff" : r"$\Delta$",
    "diff_std" : r"$\sigma$",
    "fit_diff" : r"fit$\Delta$",
    "zaswiadczenia2t" : "Zaśw.",
}

colsToKeep = [recols[c] if c in recols else c for c in colsToKeepBase]

#print ('colsToKeep', colsToKeep)

def esc(val):
    return escape_latex(str(val))

out["P"] = 0
out["X"] = 0
out["B"] = 0
out["N"] = 0
out["sum"] = 0

# Zliczamy globalną liczbę rozmaitych anomalii

probaMax = 2
for probaMin, label in THRESHOLDS:
    for direction in ['POS', 'NEG']:
        for g in fileGroups:
            ou = g['x_edge'+direction]
            ext = (ou[(ou['x_edge'] >= probaMin) &(ou['x_edge'] < probaMax)])
            #print(direction, label, globalVals[label+'g'], ext.shape[0])
            globalVals[label+'g'] += ext.shape[0]
    probaMax = probaMin
            
ncols = len(colsToKeep)

def genTables(i, v, prok):
    extracts = {'POS': {},'NEG': {}}
    tabord=0
    tabelki = ''
    probaMax = 2
    for probaMin, label in THRESHOLDS:
        for direction in ['POS', 'NEG']:
            for g in fileGroups:
                ou = g['x_edge'+direction]
                iii = (ou['x_edge'] >= probaMin) &(ou['x_edge'] < probaMax)
                if prok:
                    iii &= ou['prokuratura']==prok
                ext = (ou[iii]).copy()
                ext ['kat'] = g['kat']
                if not ext.empty:
                    if label in extracts[direction]:
                        extracts[direction][label] = pd.concat (
                            [extracts[direction][label], ext],
                            axis=0,
                            ignore_index=True)
                    else:
                        extracts[direction][label] = ext
            if label in extracts[direction]:
                n = extracts[direction][label].shape[0]
                #print ('n=', n)
                if prok:
                    out.at[i, label] += n
                    out.at[i, 'sum'] += n
                    globalVals[label+'incr'] += n
                else:
                    globalVals[label+'gAlt'] += n
                    globalVals['sumgAlt'] += n
                    globalVals[label+'incrAlt'] += n
                
                renamed = extracts[direction][label].rename(columns=recols)[colsToKeep + ['Siedziba']]
                renamed = renamed.sort_values(by="x_edge", ascending=False)
                formatters = {
                    c: (euro_fmt if pd.api.types.is_float_dtype(
                        renamed[c]) else esc)  for c in colsToKeep}
                formatters ['fitNAW'] = euro_fmt1
                formatters ['fitTRZA'] = euro_fmt1
                formatters [r'fit$\Delta$'] = euro_fmt1
                formatters [r'$\sigma$'] = euro_fmt1
                formatters ['D'] = euro_fmt1
                formatters ['diff_std'] = euro_fmt1
                formatters ['Gmina'] = split_middle
                formatters ['Powiat'] = split_middle
                escaped_header = [escape_latexNomath(str(c)) for c in renamed.columns[:-1]]
                #print ('renamed len', renamed.shape[0])
                base_text = renamed.to_latex (
                    columns=colsToKeep,
                    index=False,
                    longtable=True,
                    escape=False,
                    caption=rf"Obwody z wynikiem  {EXPLAIN[label]} na korzyść {BENEF[direction]} według modelu\label{{tabord:{tabord}}}." ,
                    label=f"tab:{direction}{label}",
                    formatters=formatters,
                    header=escaped_header,
                    column_format = "crrllrrrrrrrrrr",
                    multicolumn=False
                    
                )
                tabord += 1
                base_text = base_text.replace("Continued on next page", pl_msg)
                notes    = renamed['Siedziba'].map(escape_latex).tolist()
                #print ('len(notes)', len(notes))
                
                #print ('adresy',notes)
                note_it  = iter(notes)
                new_lines = []
                for ln in base_text.splitlines():
                    new_lines.append(ln)

                    # heuristics: data rows end with '\\' **and** contain at least one '&'
                    if ln.rstrip().endswith(r'\\') and '&' in ln:
                        if not all (sub in ln for sub in [' Teryt ', ' Gmina ', ' Nr ']):
                            note_text = next(note_it)
                            note_line = rf"\nopagebreak\multicolumn{{{ncols}}}{{l}}{{\textit{{{note_text}}}}} \\ \midrule"
                            new_lines.append(note_line)
                tabelki += '\n'.join (new_lines)
        probaMax = probaMin
    return tabelki

gtabelki = genTables(None, None, None)
    
if gtabelki:
    with open(s / 'tables.tex', "w", encoding="utf-8") as f:
        f.write(gtabelki)


for i, v in out.iterrows():
    (s / v["dir"]).mkdir (parents=True, exist_ok=True)
    tabelki = genTables (i, v, v['nazwa'])
    
    if tabelki:
        with open(s / v['dir'] / 'tables.tex', "w", encoding="utf-8") as f:
            f.write(tabelki)


if False:

    tabord=0
    extracts = {'POS': {},'NEG': {}}
    tabelki = ''
    probaMax = 2
    for probaMin, label in THRESHOLDS:
        for direction in ['POS', 'NEG']:
            for g in fileGroups:
                ou = g['x_edge'+direction]
                ext = (ou[(ou['x_edge'] >= probaMin) &(ou['x_edge'] < probaMax) & (ou['prokuratura']==v['nazwa'])]).copy()
                ext ['kat'] = g['kat']
                if not ext.empty:
                    if label in extracts[direction]:
                        extracts[direction][label] = pd.concat (
                            [extracts[direction][label], ext],
                            axis=0,
                            ignore_index=True)
                    else:
                        extracts[direction][label] = ext
            if label in extracts[direction]:
                n = extracts[direction][label].shape[0]
                #print ('n=', n)
                out.at[i, label] += n
                out.at[i, 'sum'] += n
                globalVals[label+'incr'] += n
                renamed = extracts[direction][label].rename(columns=recols)[colsToKeep + ['Siedziba']]
                formatters = {
                    c: (euro_fmt if pd.api.types.is_float_dtype(
                        renamed[c]) else esc)  for c in colsToKeep}
                formatters ['fitNAW'] = euro_fmt1
                formatters ['fitTRZA'] = euro_fmt1
                formatters [r'fit$\Delta$'] = euro_fmt1
                formatters [r'$\sigma$'] = euro_fmt1
                formatters ['D'] = euro_fmt1
                formatters ['diff_std'] = euro_fmt1
                formatters ['Gmina'] = split_middle
                formatters ['Powiat'] = split_middle
                escaped_header = [escape_latexNomath(str(c)) for c in renamed.columns[:-1]]
                #print ('renamed len', renamed.shape[0])
                base_text = renamed.to_latex (
                    columns=colsToKeep,
                    index=False,
                    longtable=True,
                    escape=False,
                    caption=rf"Obwody z wynikiem  {EXPLAIN[label]} na korzyść {BENEF[direction]} według modelu\label{{tabord:{tabord}}}." ,
                    label=f"tab:{direction}{label}",
                    formatters=formatters,
                    header=escaped_header,
                    column_format = "crrllrrrrrrrrrr",
                    multicolumn=False
                    
                )
                tabord += 1
                base_text = base_text.replace("Continued on next page", pl_msg)
                notes    = renamed['Siedziba'].map(escape_latex).tolist()
                #print ('len(notes)', len(notes))
                
                #print ('adresy',notes)
                note_it  = iter(notes)
                new_lines = []
                for ln in base_text.splitlines():
                    new_lines.append(ln)

                    # heuristics: data rows end with '\\' **and** contain at least one '&'
                    if ln.rstrip().endswith(r'\\') and '&' in ln:
                        if not all (sub in ln for sub in [' Teryt ', ' Gmina ', ' Nr ']):
                            note_text = next(note_it)
                            note_line = rf"\nopagebreak\multicolumn{{{ncols}}}{{l}}{{\textit{{{note_text}}}}} \\ \midrule"
                            new_lines.append(note_line)
                tabelki += '\n'.join (new_lines)
        probaMax = probaMin
    #if tabelki:
    #    with open(s / v['dir'] / 'tables.tex', "w", encoding="utf-8") as f:
    #        f.write(tabelki)

out.to_csv ('merge.csv', sep=';', index=True,
         header=True, encoding="utf-8",
         quoting=1, quotechar='"')

with open('doniesienie-mono.tex', encoding="utf-8") as f:
    texSrc = f.read()

globalVals['Ppr'] = rf"{globalVals['Pg']/globalVals['Gg']*100:1.3f}\%".replace(".", ",")
globalVals['Xpr'] = rf"{globalVals['Xg']/globalVals['Gg']*100:1.3f}\%".replace(".", ",")
globalVals['Bpr'] = rf"{globalVals['Bg']/globalVals['Gg']*100:1.3f}\%".replace(".", ",")
globalVals['Npr'] = rf"{globalVals['Ng']/globalVals['Gg']*100:1.3f}\%".replace(".", ",")

#f"{x:.6f}".replace(".", ",")

for i, v in out.iterrows():
    res = texSrc
    for col in out.columns:
        res = res.replace (f"===={col}====", str(v[col]))
    for k in globalVals:
        res = res.replace (f"===={k}====", str(globalVals[k]))
    with open (s / v['dir'] / 'doniesienie.tex', "w", encoding="utf-8") as f:
        f.write(res)


with open('doniesienie-multi.tex', encoding="utf-8") as f:
    texMultiSrc = f.read()
res = texMultiSrc
for k in globalVals:
    res = res.replace (f"===={k}====", str(globalVals[k]))
with open (s / 'doniesienie-multi-processed.tex', "w", encoding="utf-8") as f:
    f.write(res)


        
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
