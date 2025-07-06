python lincyferki-prepare.py

cat > /dev/null <<EOF

bierze domyślnie:
   protokoly_po_obwodach_utf8-fixed.xlsx
   protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx
   commission_combined.xlsx
(ten trzeci, to składy komisji według komitetów)

i produkuje kilka Y<nazwa>D2025.xlsx -- regresja i dane statystyczne
dla różnych grup komisji

Nie ma restrykcji ze względu na typ obwodu
Nie ma restrykcji ze względu na liczbę ludnosci gminy

EOF

python lincyferki-merge.py

cat > /dev/null <<EOF

Neeaktualny. Miał złączać wynik lincyferki-prepare.py z
commission_combined.xlsx
czyli ze składami komisji po komitetach wyborczych.

Obecnie zbędne, bo lincyferki-prepare.py robi to złączenie
EOF

python lincyferki-display.py

cat > /dev/null <<EOF

Zasadniczo zmieniam logikę:
nazwa pliku we jet pierwszym argumentem

-H: histogramy główne
-c <arg>: cyferki dla kasy <arg> <arg> może być all lub red lub blue lub black
-o: generuj tabelę outliers, której nazwa, to outliers-<nazwa pliku we>
-l <int>: z podziałem na ludność gminy ponizej i powyżej <int>. Raczej nie używamy, bo podział według ludności robiony jest wcześniej, w lincyferki-prepare.py

-W: oddizelnie wyliczaj Warszawę (obecnie niezalecane)
-w: podział na województwa (obecnie niezalecane)
-m: nie wiem, co to
-y <rok>: rok. W domysle 2025. Dopuszczalne wartości: 2020, 2025
EOF
