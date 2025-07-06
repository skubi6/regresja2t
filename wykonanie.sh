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
