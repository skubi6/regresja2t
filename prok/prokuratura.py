import pandas as pd
import re

prokuraturyO = r"""§ 3. Ustala się siedziby i obszary właściwości prokuratur okręgowych:
1) w obszarze właściwości Prokuratury Regionalnej w Białymstoku:
a) Prokuraturę Okręgową w Białymstoku – obejmującą obszar właściwości Prokuratur Rejonowych: Białystok-Po-
łudnie w Białymstoku, Białystok-Północ w Białymstoku i w: Białymstoku, Bielsku Podlaskim, Hajnówce, Sie-
miatyczach i Sokółce,
b) Prokuraturę Okręgową w Łomży – obejmującą obszar właściwości Prokuratur Rejonowych w: Grajewie, Kolnie,
Łomży, Wysokiem Mazowieckiem i Zambrowie,
c) Prokuraturę Okręgową w Olsztynie – obejmującą obszar właściwości Prokuratur Rejonowych w: Bartoszycach,
Biskupcu, Giżycku, Kętrzynie, Lidzbarku Warmińskim, Mrągowie, Nidzicy, Olsztyn-Południe w Olsztynie, Olsz-
tyn-Północ w Olsztynie, Piszu i Szczytnie,
d) Prokuraturę Okręgową w Ostrołęce – obejmującą obszar właściwości Prokuratur Rejonowych w: Ostrołęce,
Ostrowi Mazowieckiej, Przasnyszu, Pułtusku i Wyszkowie,
e) Prokuraturę Okręgową w Suwałkach – obejmującą obszar właściwości Prokuratur Rejonowych w: Augustowie,
Ełku, Olecku, Sejnach i Suwałkach;
2) w obszarze właściwości Prokuratury Regionalnej w Gdańsku:
a) Prokuraturę Okręgową w Bydgoszczy – obejmującą obszar właściwości Prokuratur Rejonowych: Bydgoszcz-Pół-
noc w Bydgoszczy, Bydgoszcz-Południe w Bydgoszczy i w: Inowrocławiu, Mogilnie, Nakle n. Notecią, Szubinie,
Świeciu i Tucholi,
b) Prokuraturę Okręgową w Elblągu – obejmującą obszar właściwości Prokuratur Rejonowych w: Braniewie, Dział-
dowie, Elblągu, Iławie, Nowym Mieście Lubawskim i Ostródzie,
c) Prokuraturę Okręgową w Gdańsku – obejmującą obszar właściwości Prokuratur Rejonowych: Gdańsk-Oliwa
w Gdańsku, Gdańsk-Śródmieście w Gdańsku, Gdańsk-Wrzeszcz w Gdańsku i w: Gdyni, Kartuzach, Kościerzynie,
Kwidzynie, Malborku, Pruszczu Gdańskim, Pucku, Sopocie, Starogardzie Gdańskim, Tczewie i Wejherowie,
d) Prokuraturę Okręgową w Słupsku – obejmującą obszar właściwości Prokuratur Rejonowych w: Bytowie, Chojni-
cach, Człuchowie, Lęborku, Miastku i Słupsku,
e) Prokuraturę Okręgową w Toruniu – obejmującą obszar właściwości Prokuratur Rejonowych w: Brodnicy, Chełmnie,
Golubiu-Dobrzyniu, Grudziądzu, Toruń Centrum-Zachód w Toruniu, Toruń-Wschód w Toruniu i Wąbrzeźnie,
f) Prokuraturę Okręgową we Włocławku – obejmującą obszar właściwości Prokuratur Rejonowych w: Aleksandro-
wie Kujawskim, Lipnie, Radziejowie, Rypinie i Włocławku;
3) w obszarze właściwości Prokuratury Regionalnej w Katowicach:
a) Prokuraturę Okręgową w Bielsku-Białej – obejmującą obszar właściwości Prokuratur Rejonowych: Biel-
sko-Biała-Południe w Bielsku-Białej, Bielsko-Biała-Północ w Bielsku-Białej, w Cieszynie i Żywcu,
b) Prokuraturę Okręgową w Częstochowie – obejmującą obszar właściwości Prokuratur Rejonowych: Często-
chowa-Południe w Częstochowie, Częstochowa-Północ w Częstochowie i w: Częstochowie, Lublińcu i Myszkowie,
c) Prokuraturę Okręgową w Gliwicach – obejmującą obszar właściwości Prokuratur Rejonowych: Gliwice-Wschód
w Gliwicach, Gliwice-Zachód w Gliwicach i w: Jastrzębiu-Zdroju, Raciborzu, Rudzie Śląskiej, Rybniku, Tarnow-
skich Górach, Wodzisławiu Śląskim, Zabrzu i Żorach,
Dziennik Ustaw – 4 – Poz. 415
d) Prokuraturę Okręgową w Katowicach – obejmującą obszar właściwości Prokuratur Rejonowych w: Bytomiu, Cho-
rzowie, Katowice-Południe w Katowicach, Katowice-Północ w Katowicach, Katowice-Wschód w Katowicach,
Katowice-Zachód w Katowicach, Mikołowie, Mysłowicach, Pszczynie, Siemianowicach Śląskich i Tychach,
e) Prokuraturę Okręgową w Sosnowcu – obejmującą obszar właściwości Prokuratur Rejonowych w: Będzinie, Dą-
browie Górniczej, Jaworznie, Sosnowiec-Południe w Sosnowcu, Sosnowiec-Północ w Sosnowcu i Zawierciu;
4) w obszarze właściwości Prokuratury Regionalnej w Krakowie:
a) Prokuraturę Okręgową w Kielcach – obejmującą obszar właściwości Prokuratur Rejonowych w: Busku-Zdroju,
Jędrzejowie, Kielce-Wschód w Kielcach, Kielce-Zachód w Kielcach, Końskich, Opatowie, Ostrowcu Świętokrzy-
skim, Pińczowie, Sandomierzu, Skarżysku-Kamiennej, Starachowicach, Staszowie i Włoszczowie,
b) Prokuraturę Okręgową w Krakowie – obejmującą obszar właściwości Prokuratur Rejonowych w: Chrzanowie,
Kraków-Krowodrza w Krakowie, Kraków-Nowa Huta w Krakowie, Kraków-Podgórze w Krakowie, Kra-
ków-Prądnik Biały w Krakowie, Kraków-Śródmieście Wschód w Krakowie, Kraków-Śródmieście Zachód w Kra-
kowie, Miechowie, Myślenicach, Olkuszu, Oświęcimiu, Suchej Beskidzkiej, Wadowicach i Wieliczce,
c) Prokuraturę Okręgową w Nowym Sączu – obejmującą obszar właściwości Prokuratur Rejonowych w: Gorlicach,
Limanowej, Muszynie, Nowym Sączu, Nowym Targu i Zakopanem,
d) Prokuraturę Okręgową w Tarnowie – obejmującą obszar właściwości Prokuratur Rejonowych w: Bochni, Brzesku,
Dąbrowie Tarnowskiej i Tarnowie;
5) w obszarze właściwości Prokuratury Regionalnej w Lublinie:
a) Prokuraturę Okręgową w Lublinie – obejmującą obszar właściwości Prokuratur Rejonowych w: Białej Podlaskiej,
Chełmie, Kraśniku, Lubartowie, Lublin-Południe w Lublinie, Lublin-Północ w Lublinie, Lublinie, Łukowie,
Opolu Lubelskim, Parczewie, Puławach, Radzyniu Podlaskim, Rykach, Świdniku i Włodawie,
b) Prokuraturę Okręgową w Radomiu – obejmującą obszar właściwości Prokuratur Rejonowych w: Grójcu, Kozie-
nicach, Lipsku, Przysusze, Radom-Wschód w Radomiu, Radom-Zachód w Radomiu i Zwoleniu,
c) Prokuraturę Okręgową w Siedlcach – obejmującą obszar właściwości Prokuratur Rejonowych w: Garwolinie,
Mińsku Mazowieckim, Siedlcach, Sokołowie Podlaskim i Węgrowie,
d) Prokuraturę Okręgową w Zamościu – obejmującą obszar właściwości Prokuratur Rejonowych w: Biłgoraju, Hru-
bieszowie, Janowie Lubelskim, Krasnymstawie, Tomaszowie Lubelskim i Zamościu;
6) w obszarze właściwości Prokuratury Regionalnej w Łodzi:
a) Prokuraturę Okręgową w Łodzi – obejmującą obszar właściwości Prokuratur Rejonowych w: Brzezinach, Kutnie,
Łęczycy, Łowiczu, Łódź-Bałuty w Łodzi, Łódź-Górna w Łodzi, Łódź-Polesie w Łodzi, Łódź-Śródmieście w Ło-
dzi, Łódź-Widzew w Łodzi, Pabianicach, Rawie Mazowieckiej, Skierniewicach i Zgierzu,
b) Prokuraturę Okręgową w Ostrowie Wielkopolskim – obejmującą obszar właściwości Prokuratur Rejonowych w:
Jarocinie, Kaliszu, Kępnie, Krotoszynie, Ostrowie Wielkopolskim, Ostrzeszowie i Pleszewie,
c) Prokuraturę Okręgową w Piotrkowie Trybunalskim – obejmującą obszar właściwości Prokuratur Rejonowych w:
Bełchatowie, Opocznie, Piotrkowie Trybunalskim, Radomsku i Tomaszowie Mazowieckim,
d) Prokuraturę Okręgową w Płocku – obejmującą obszar właściwości Prokuratur Rejonowych w: Ciechanowie, Go-
styninie, Mławie, Płocku, Płońsku, Sierpcu, Sochaczewie i Żyrardowie,
e) Prokuraturę Okręgową w Sieradzu – obejmującą obszar właściwości Prokuratur Rejonowych w: Łasku, Poddębi-
cach, Sieradzu, Wieluniu i Zduńskiej Woli;
7) w obszarze właściwości Prokuratury Regionalnej w Poznaniu:
a) Prokuraturę Okręgową w Koninie – obejmującą obszar właściwości Prokuratur Rejonowych w: Kole, Koninie,
Słupcy i Turku,
b) Prokuraturę Okręgową w Poznaniu – obejmującą obszar właściwości Prokuratur Rejonowych w: Chodzieży,
Gnieźnie, Gostyniu, Grodzisku Wielkopolskim, Kościanie, Lesznie, Nowym Tomyślu, Obornikach, Pile, Po-
znań-Grunwald w Poznaniu, Poznań-Nowe Miasto w Poznaniu, Poznań-Stare Miasto w Poznaniu, Poznań-Wilda
w Poznaniu, Rawiczu, Szamotułach, Śremie, Środzie Wielkopolskiej, Trzciance, Wągrowcu, Wolsztynie, Wrześni
i Złotowie,
c) Prokuraturę Okręgową w Zielonej Górze – obejmującą obszar właściwości Prokuratur Rejonowych w: Krośnie
Odrzańskim, Nowej Soli, Świebodzinie, Wschowie, Zielonej Górze, Żaganiu i Żarach;
Dziennik Ustaw – 5 – Poz. 415
8) w obszarze właściwości Prokuratury Regionalnej w Rzeszowie:
a) Prokuraturę Okręgową w Krośnie – obejmującą obszar właściwości Prokuratur Rejonowych w: Brzozowie, Jaśle,
Krośnie, Lesku i Sanoku,
b) Prokuraturę Okręgową w Przemyślu – obejmującą obszar właściwości Prokuratur Rejonowych w: Jarosławiu, Lu-
baczowie, Przemyślu i Przeworsku,
c) Prokuraturę Okręgową w Rzeszowie – obejmującą obszar właściwości Prokuratur Rejonowych w: Dębicy, Leżaj-
sku, Łańcucie, Ropczycach, Rzeszowie dla miasta Rzeszów, w Rzeszowie i Strzyżowie,
d) Prokuraturę Okręgową w Tarnobrzegu – obejmującą obszar właściwości Prokuratur Rejonowych w: Kolbuszowej,
Mielcu, Nisku, Stalowej Woli i Tarnobrzegu;
9) w obszarze właściwości Prokuratury Regionalnej w Szczecinie:
a) Prokuraturę Okręgową w Gorzowie Wielkopolskim – obejmującą obszar właściwości Prokuratur Rejonowych w:
Gorzowie Wielkopolskim, Międzyrzeczu, Słubicach, Strzelcach Krajeńskich i Sulęcinie,
b) Prokuraturę Okręgową w Koszalinie – obejmującą obszar właściwości Prokuratur Rejonowych w: Białogardzie,
Drawsku Pomorskim, Kołobrzegu, Koszalinie, Sławnie, Szczecinku i Wałczu,
c) Prokuraturę Okręgową w Szczecinie – obejmującą obszar właściwości Prokuratur Rejonowych w: Choszcznie,
Goleniowie, Gryficach, Gryfinie, Kamieniu Pomorskim, Łobzie, Myśliborzu, Pyrzycach, Stargardzie, Szcze-
cin-Niebuszewo w Szczecinie, Szczecin-Prawobrzeże w Szczecinie, Szczecin-Śródmieście w Szczecinie, Szcze-
cin-Zachód w Szczecinie i Świnoujściu;
10) w obszarze właściwości Prokuratury Regionalnej w Warszawie:
a) Prokuraturę Okręgową w Warszawie – obejmującą obszar właściwości prokuratur rejonowych w: Grodzisku Ma-
zowieckim, Piasecznie, Pruszkowie, Warszawa-Mokotów w Warszawie, Warszawa-Ochota w Warszawie, War-
szawa-Śródmieście w Warszawie, Warszawa Śródmieście-Północ w Warszawie, Warszawa-Ursynów w Warsza-
wie, Warszawa-Wola w Warszawie i Warszawa-Żoliborz w Warszawie,
b) Prokuraturę Okręgową Warszawa-Praga w Warszawie – obejmującą obszar właściwości Prokuratur Rejonowych
w: Legionowie, Nowym Dworze Mazowieckim, Otwocku, Warszawa-Praga Południe w Warszawie, War-
szawa-Praga Północ w Warszawie i Wołominie;
11) w obszarze właściwości Prokuratury Regionalnej we Wrocławiu:
a) Prokuraturę Okręgową w Jeleniej Górze – obejmującą obszar właściwości Prokuratur Rejonowych w: Bolesławcu,
Jeleniej Górze, Kamiennej Górze, Lubaniu, Lwówku Śląskim i Zgorzelcu,
b) Prokuraturę Okręgową w Legnicy – obejmującą obszar właściwości Prokuratur Rejonowych w: Głogowie, Jawo-
rze, Legnicy, Lubinie i Złotoryi,
c) Prokuraturę Okręgową w Opolu – obejmującą obszar właściwości Prokuratur Rejonowych w: Brzegu, Głubczy-
cach, Kędzierzynie-Koźlu, Kluczborku, Nysie, Oleśnie, Opolu, Prudniku i Strzelcach Opolskich,
d) Prokuraturę Okręgową w Świdnicy – obejmującą obszar właściwości Prokuratur Rejonowych w: Bystrzycy
Kłodzkiej, Dzierżoniowie, Kłodzku, Świdnicy, Wałbrzychu i Ząbkowicach Śląskich,
e) Prokuraturę Okręgową we Wrocławiu – obejmującą obszar właściwości Prokuratur Rejonowych w: Miliczu, Oleś-
nicy, Oławie, Strzelinie, Środzie Śląskiej, Trzebnicy, Wołowie, Wrocław-Fabryczna we Wrocławiu, Wro-
cław-Krzyki Wschód we Wrocławiu, Wrocław-Krzyki Zachód we Wrocławiu, Wrocław-Psie Pole we Wrocławiu,
Wrocław-Stare Miasto we Wrocławiu i Wrocław-Śródmieście we Wrocławiu.
"""


prokuraturyR = r"""



# Wczytaj dane komisji
df = pd.read_excel('outlierskraj2025-prokuratura.xlsx')
# Tekst rozporządzenia 2025 (skrócony wyciąg z Dziennika Ustaw)
rozporzadzenie_text = """
... (tutaj wstawiamy tekst rozporządzenia lub jego istotne fragmenty, np. zawartość PDF) ...
"""

# Słownik prokuratur okręgowych z adresami (uzupełniony na podstawie BIP)
okręg_adres = {
    "Prokuratura Okręgowa w Białymstoku": "ul. Mickiewicza 9, 15-213 Białystok",
    "Prokuratura Okręgowa we Włocławku": "ul. Orla 2, 87-800 Włocławek",
    # ... (pozostałe okręgi)
}

# Normalizacja nazw gmin
def normalize_gmina(name):
    name = name.replace("gm. ", "").replace("m. ", "")
    return name

df['Gmina_clean'] = df['Gmina'].apply(normalize_gmina)

# Specjalne mapowanie dla dzielnic Warszawy
warszawa_lewobrzeżna = {"Bemowo","Bielany","Mokotów","Ochota","Śródmieście","Ursus","Ursynów","Wilanów","Włochy","Żoliborz","Wola"} 
warszawa_praga = {"Białołęka","Praga-Północ","Praga-Południe","Rembertów","Targówek","Wawer","Wesoła","Wilanów","Wesoła"} 

results = []
for _, row in df.iterrows():
    gmina = row['Gmina_clean']
    # Sprawdź Warszawę
    if gmina in warszawa_lewobrzeżna:
        okreg = "Prokuratura Okręgowa w Warszawie"
    elif gmina in warszawa_praga:
        okreg = "Prokuratura Okręgowa Warszawa-Praga w Warszawie"
    else:
        # Wyszukaj w tekście rozporządzenia wzmianki o gminie
        pattern = rf"\b{re.escape(gmina)}\b"
        match = re.search(rf"(\d+\)) w obszarze właściwości Prokuratury Okręgowej w ([^:]+):(?:(?!\d+\)).)*\b{re.escape(gmina)}\b", rozporzadzenie_text, flags=re.DOTALL)
        if match:
            okreg = "Prokuratura Okręgowa w " + match.group(2)
        else:
            okreg = None
    # Pobierz adres
    adres = okręg_adres.get(okreg, "")
    results.append((okreg, adres))

df[['Prokuratura Okręgowa', 'Adres']] = pd.DataFrame(results, columns=['Prokuratura Okręgowa','Adres'])
# Dodaj kolumnę z linkiem do źródła (tu: Dziennik Ustaw poz. 415/2025)
df['Źródło'] = "Dz.U. 2025 poz. 415 (tabela właściwości)"
df.to_excel('komisje_z_prokuraturami.xlsx', index=False)
print(df[['Gmina','Prokuratura Okręgowa','Adres','Źródło']].head(10))
