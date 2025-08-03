import pdfplumber
import pandas as pd
import numpy as np
import re
from pdfplumber.utils import cluster_objects

patternLp = re.compile(r'^_?\d{1,3}\*?$')
patternNrKom = re.compile(r'\d{1,3}$')
pageNum = re.compile(r'^\d\d$')

def isLp(s: str) -> bool:
    return bool(patternLp.fullmatch(s))

def chars_in_text_order(page):
    # same ordering used internally by extract_text()
    return sorted(page.chars, key=lambda c: (c["doctop"], c["x0"]))




t = []
t2 = []
tWithPos = []
with pdfplumber.open('Toroj-OCR.pdf') as pdf:
    for i in range(17, 21):
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

torojKolumns = ["tabl#", "lp-toroj", "nr", "gmina", "adr", "uprawnieni2t", "Dtrza%", "Dnaw%",
                               "Dtrza", "Dnaw", "zm zaswiadczenia", "ogledziny"]

#torojLok6 = [104, 160, 385, 424, 470, 522, 586, 641, 662, 682, 5000]
torojLok6 = [104, 176, 385, 424, 470, 522, 586, 682, 712, 5000]
torojLok7 = [104, 178, 440, 480, 512, 552, 586, 642, 5000]
#             0    1    2    3    4    5    6    7  8
torojT = pd.DataFrame(columns=torojKolumns)

#ADR_Xmin = 172

lpToroj = 1

for p in tWithPos:
    lines = splitList (p, '\n')
    for lList in lines:
        l = toText(lList)
        #print ('linia', l)
        if 'Ranking obwodów wg anomalii' in l:
            if "Tabela 6." in l:
                tablNum=1
                torojLok = torojLok6
            elif "Tabela 7." in l:
                tablNum=2
                torojLok = torojLok7
            else:
                print ("Bad text", l)
                sys.exit(1)
            continue
        if "Nr OKW " in l or "NrOKW " in l:
            ogonGm = []
            ogonAdr = []
            continue
        textFrags = [[]] + [[] for e in torojLok]
        curSection = 0
        for obj in lList:
            if not isinstance (obj, str):
                while torojLok[curSection] <= obj["x1"]:
                    curSection += 1
                textFrags[curSection].append(obj)
        strFrags = [toText(e) for e in textFrags]
        for i, e in enumerate (strFrags):
            print (f"{i} <{e}> ", end='')
        print()
        strFrags = [e.strip() for e in strFrags]
        if not ('' != strFrags[1] or ('' != strFrags[2] and not bool(pageNum.fullmatch(strFrags[2])))):
            ogonGm = []
            ogonAdr = []
            continue
        if '' == strFrags[0]:
            if '' != strFrags[1]:
                ogonGm.append(strFrags[1])
            if '' != strFrags[2]:
                ogonAdr.append(strFrags[2])
            print (f"ogony <{strFrags[1]}> <{strFrags[2]}>")
            continue
        
        row = {}
        row['tabl#'] = tablNum
        row['lp-toroj'] = lpToroj
        lpToroj += 1
        row['nr'] = int(strFrags[0])

        ogonGm.append(strFrags[1])
        gmina = ogonGm[0]
        print ('ogon', ogonGm, 'gmina', gmina)
        for e in ogonGm[1:]:
            if '-' == gmina[-1]:
                gmina += e
            else:
                gmina += ' ' + e
        if "rr. Bielsko-Biała" == gmina:
            gmina = "m. Bielsko-Biała"
        if "m.Łeba" == gmina:
            gmina = "m. Łeba"
        row['gmina'] = gmina
        ogonAdr.append(strFrags[2])
        addr = ogonAdr[0]
        for e in ogonAdr[1:]:
            if '-' == addr[-1]:
                addr += e
            else:
                addr += ' ' + e
        row['adr'] = addr
        print (f'gmina <{gmina}> adr<{addr}>')

        for i, e in enumerate (strFrags):
            print (f'{i} <{e}>', end = ' ')
        print ()
        
        row['uprawnieni2t'] = int(strFrags[3])
        row['Dtrza%'] = float(strFrags[4].replace(",", "."))
        row['Dnaw%'] = float(strFrags[5].replace(",", "."))
        row['Dtrza'] = float(strFrags[6].replace(",", "."))
        if "96t41" == strFrags[7]:
            print ('fix[7]', strFrags[7])
            strFrags[7] = "96,41"
        row['Dnaw'] = float(strFrags[7].replace(",", "."))
        row['zm zaswiadczenia'] = float(strFrags[8].replace(",", "."))
        row['ogledziny'] = 'TAK' == strFrags[9]
        
        ogonGm = []
        ogonAdr = []
        torojT = pd.concat ([torojT, pd.DataFrame([row])], ignore_index=True)
        
torojT.to_excel("toroj-converted.xlsx", index=False)
