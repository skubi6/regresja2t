import pdfplumber
import pandas as pd
import numpy as np
import re

patternLp = re.compile(r'^_?\d{1,3}\*?$')
patternNrKom = re.compile(r'\d{1,3}$')

def isLp(s: str) -> bool:
    return bool(patternLp.fullmatch(s))

t = []
with pdfplumber.open('haman-OCR.pdf') as pdf:
    for i in range(35):
        page = pdf.pages[i]
        t.append(page.extract_text())
        tables = page.extract_table()

ogon = []
lastNum = 0
tablNum = 0

hamanT = pd.DataFrame(columns=["tabl#", "lp-haman", "star", "gmina", "nr",
                               "adr", "karty wazne", "wazne", "Naw", "Trza",
                               "Naw%", "Trza%",
                               "fitNaw%", "fitTrza%", "blad%", "blad/sigma",
                               "blad", "odwr"])

for p in t:
    lines = p.split('\n')
    for l in lines:
        #print (l)
        if ('Siedziba komisji' in l
            or 'a in e c ó r' in l
            or 'o d w r ó c e ni a' in l
            or '......' in l
            or ' a s o g w' in l
            or 'ł' == l
            or '[' == l):
            
            ogon = []
            continue
        words = l.split(' ')
        if 10 < len(words) and isLp (words [0]):
            row = {}
            lpStr = words[0]
            if '_' == lpStr[0]:
                lpStr = lpStr[1:]
            gwiazdka = '*' == lpStr[-1]
            if gwiazdka:
                lpStr = lpStr[:-1]
            row['lp-haman'] = int(lpStr)
            odwrocenie = False
            if 'tak' == words[-1]:
                odwrocenie = True
                words = words[:-1]
            elif 'tak' == words[-1][-3:]:
                odwrocenie = True
                words = words[:-1]
            row['odwr'] = odwrocenie
            row['blad'] = int(words[-1])
            row['blad/sigma'] = float(words[-2].replace(",", "."))
            row['blad%'] = float (words[-3].replace("%", "").replace(",", ".")) / 100
            row['fitTrza%'] = float (words[-4].replace("%", "").replace(",", ".")) / 100
            row['fitNaw%'] = float (words[-5].replace("%", "").replace(",", ".")) / 100
            row['Trza%'] = float (words[-6].replace("%", "").replace(",", ".")) / 100
            row['Naw%'] = float (words[-7].replace("%", "").replace(",", ".")) / 100
            
            row['Trza'] = int (words[-8])
            row['Naw'] = int (words[-9])
            row['wazne'] = int (words[-10])
            row['karty wazne'] = int (words[-11])
            
            
            
            nrKomIndex = 0
            for i in range (2,6):
                if patternNrKom.fullmatch(words[i]):
                    #and not patternNrKom.fullmatch(words[i+1])
                    nrKomIndex = i
                    break
            row['nr'] = int(words [nrKomIndex])
            row['gmina'] = ' '.join(words[1:nrKomIndex])
            addrL = ogon + words[nrKomIndex+1:-11]
            addr = addrL[0]
            for e in addrL[1:]:
                if '-' == addr[-1]:
                    addr += e
                else:
                    addr += ' ' + e
            
            row['adr'] = addr
            if not ((row['lp-haman'] == 1 and 10 < lastNum) or lp == lastNum+1):
                print ('WRONG ORDER')
            if row['lp-haman'] == 1:
                tablNum += 1
            row['tabl#'] = tablNum
            hamanT.concat ([hamanT, row], ignore_index=True)
            
            print (f'lp {row['lp']} *{gwiazdka} gmina<{gmina}> nrInd {nrKomIndex} addr <{addr}>'
                   f'nr {nrKom} odwr {odwrocenie} RESZTA {words [1:]}')
            print (ogon)
            ogon = []
            lastNum = lp
        else:
            ogon.append (l)
