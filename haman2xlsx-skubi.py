import pdfplumber
import pandas as pd
import numpy as np
import re
from pdfplumber.utils import cluster_objects

patternLp = re.compile(r'^_?\d{1,3}\*?$')
patternNrKom = re.compile(r'\d{1,3}$')

def isLp(s: str) -> bool:
    return bool(patternLp.fullmatch(s))

def chars_in_text_order(page):
    # same ordering used internally by extract_text()
    return sorted(page.chars, key=lambda c: (c["doctop"], c["x0"]))




#with open ("haman-fixed.txt", "r", encoding="utf-8") as f:
#    t = f.read()

t = []
t2 = []
tWithPos = []
with pdfplumber.open('haman-OCR.pdf') as pdf:
    for i in range(11, 35):
        page = pdf.pages[i]
        t.append(page.extract_text())
        rows = cluster_objects(page.chars, "doctop", tolerance=3)
        for row in rows:                      # one physical line
            rowSorted = sorted(row, key=lambda c: c["x0"])
            #        chars = chars_in_text_order(page)
            txt_parts = []
            idx_map   = []            # keeps → (char_dict OR None if spacer)
            last = None
            for ch in rowSorted:
            # inject spaces / newlines with rules similar to extract_text()
                if last is not None:
                    ## newline?
                    #if ch["doctop"] - last["doctop"] > 3:     # y_tolerance default = 3
                    #    txt_parts.append("\n");
                    #    idx_map.append("\n")
                    # space?
                    if ch["x0"] - last["x1"] > 3 and (
                            not " " == txt_parts[-1][-1]):            # x_tolerance default = 3
                        #print (f"space after <{txt_parts}>")
                        txt_parts.append(" ")
                        idx_map.append(" ")
                txt_parts.append(ch["text"])
                idx_map.append(ch)
                last = ch
            txt_parts.append("\n")
            idx_map.append("\n")
            #text_string = "".join(txt_parts)
            #t2.append (text_string)
            tWithPos.append (idx_map)
        

        # 2) demo: get the bbox of the *n*-th non-space char
        #n = 10
        #char_obj = idx_map[n]
        #print("Char #10 =", text_string[n], "at", (char_obj["x0"], char_obj["top"]))

ogonGm = []
ogonAdr = []
lastNum = 0
tablNum = 0

def splitList (bigList, val):
    res = []
    frag = []
    for e in bigList:
        if e==val:
            res.append(frag)
            frag = []
        else:
            frag.append(e)
    if 0 < len(frag):
        res.append(frag)
    return (res)

def toText (list):
    return ''.join([(ch if isinstance (ch, str) else ch["text"]) for ch in list])

#def toText (list):
#    return [(ch if isinstance (ch, str) else ch["text"]) for ch in list]

hamanT = pd.DataFrame(columns=["tabl#", "lp-haman", "star", "gmina", "nr",
                               "adr", "karty wazne", "wazne", "Naw", "Trza",
                               "Naw%", "Trza%",
                               "fitNaw%", "fitTrza%", "blad%", "blad/sigma",
                               "blad", "odwr"])

ADR_Xmin = 172

for p in tWithPos:
    lines = splitList (p, '\n')
    for lList in lines:
        l = toText(lList)
        #print ('linia', l)
        if ('Siedziba komisji' in l
            or 'a in e c ó r' in l
            or 'o d w r ó c e ni a' in l
            or '......' in l
            or ' a s o g w' in l
            or 'ł' == l
            or '[' == l):

            ogonGm = []
            ogonAdr = []
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
            row['star'] = gwiazdka
            row['lp-haman'] = int(lpStr)
            odwrocenie = False
            if 'tak' == words[-1]:
                odwrocenie = True
                words = words[:-1]
            elif 'tak' == words[-1][-3:]:
                odwrocenie = True
                words[-1] = words[-1][:-3]
            row['odwr'] = odwrocenie
            #print (l)
            #print (words)
            if 'Leśna' == words[2] and 15 == len (words):
                words.append("-17")
                print ('fixed:', words)
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
            if "|" == words[-11][-1]:
                words[-11] = words[-11][:-1]
                print ('fixed:', words)
            row['karty wazne'] = int (words[-11])



            nrKomIndex = 0
            for i in range (2,6):
                if patternNrKom.fullmatch(words[i]):
                    #and not patternNrKom.fullmatch(words[i+1])
                    nrKomIndex = i
                    break
            row['nr'] = int(words [nrKomIndex])
            gminaL = ogonGm + words[1:nrKomIndex]
            gmina = gminaL[0]
            for e in gminaL[1:]:
                if '-' == gmina[-1]:
                    gmina += e
                else:
                    gmina += ' ' + e
            row['gmina'] = gmina
            addrL = ogonAdr + words[nrKomIndex+1:-11]
            addr = addrL[0]
            for e in addrL[1:]:
                if '-' == addr[-1]:
                    addr += e
                else:
                    addr += ' ' + e
            row['adr'] = addr
            print (f'gmina <{gmina}> adr<{addr}>')
            
            if not ((row['lp-haman'] == 1 and 10 < lastNum) or row['lp-haman'] == lastNum+1):
                print ('WRONG ORDER')
            if row['lp-haman'] == 1:
                tablNum += 1
            row['tabl#'] = tablNum
            hamanT = pd.concat ([hamanT, pd.DataFrame([row])], ignore_index=True)

            #print (f'lp {row['lp-haman']} *{gwiazdka} gmina<{row['gmina']}> nrInd {nrKomIndex} addr <{addr}>'
            #       f'nr {row['nr']} odwr {odwrocenie} RESZTA {words [1:]}')
            #print (ogon)
            ogonAdr = []
            ogonGm = []
            lastNum = row['lp-haman']
        else:
            #print (lList[0], l)
            cutIndex = -1
            for i, obj in enumerate (lList):
                if isinstance (obj, str):
                    continue
                if ADR_Xmin < obj["x1"]:
                    cutIndex = i
                    break
            if -1 == cutIndex:
                ogonGm.append (l)
            elif 0 == cutIndex:
                ogonAdr.append (l)
            else:
                ogonGm.append(toText(lList[:cutIndex]).strip())
                ogonAdr.append(toText(lList[cutIndex:]).strip())

hamanT.to_excel("haman-converted.xlsx", index=False)
