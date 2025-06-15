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

KEY1, KEY2 = "Teryt Gminy", "Nr komisji"
c1, c2 = "NAWROCKI Karol Tadeusz", "TRZASKOWSKI Rafał Kazimierz"

DATA_DIR = Path(".")
EXTRA = pd.read_excel(DATA_DIR / "commission_combined.xlsx")
EXTRA.rename(columns={"Nr obw.": "Nr komisji", "TERYT gminy" : "Teryt Gminy"}, inplace=True)

dup_mask  = EXTRA.duplicated([KEY1, KEY2], keep=False)
dup_keys  = EXTRA.loc[dup_mask, [KEY1, KEY2]].drop_duplicates()
n_dupes   = len(dup_keys)

def mergeY (label):
    Y = pd.read_excel(DATA_DIR / f"Y{label}.xlsx")

    Y = Y.merge(
        EXTRA,
        #left_index=True,
        on=[KEY1, KEY2],
        how="left",
        indicator=True,
        suffixes=("", "_extra"),
    )

    writerY = pd.ExcelWriter(f"Y{label}-merged.xlsx", engine="xlsxwriter")
    Y.to_excel(writerY, sheet_name="Y", index=False)

    writerY.close()

for l in ["", "miasta", "wies", "zagranica"]:
    print('label', l)
    mergeY(l)
nowAfterInit = datetime.now()
print (nowAfterInit-nowStart)
