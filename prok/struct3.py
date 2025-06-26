
"""hierarchy_parser.py – strict parser for §-structured statute text.

Updated 2025-06-26 — complete, executable version
------------------------------------------------
* Strict mode – any mismatch raises `ValueError` and exits non-zero.
* § 3 flattened – its a)-, b)-… clauses are stored directly on `Part`.
* Head helpers – `split_head_tail`, `normalize_head`.
* Para 3 ▸ subpart 2 tail logic – cleans and splits tails into lists.

Run:
    python hierarchy_parser.py /path/to/prokuratura.txt | less
"""

from __future__ import annotations
import pandas as pd

import pprint
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
# ---------------------------------------------------------------------------
# Regex patterns anchored at line starts
# ---------------------------------------------------------------------------
PART_RE = r"^§\s*\d+\."
SUBPART_RE = r"^\s*\d+\)"
SUBSUBPART_RE = r"^\s*[a-z]\)"           # strict: single-letter only

# ---------------------------------------------------------------------------
# Segmentation helper – returns contiguous chunks starting with *pattern*
# ---------------------------------------------------------------------------
def _segment(pattern: str, src: str) -> List[str]:
    """Return slices that *start* with *pattern* (strict).

    Raises `ValueError` if the first line doesn’t match *pattern* or if *src*
    contains no matches at all. Prevents silent misalignment.
    """
    matches = list(re.finditer(pattern, src, flags=re.M))
    if not matches or matches[0].start() != 0:
        raise ValueError(f"Chunk does not start with expected header {pattern!r}.")

    bounds = [m.start() for m in matches] + [len(src)]
    return [src[bounds[i] : bounds[i + 1]] for i in range(len(matches))]

# ---------------------------------------------------------------------------
# Helper utilities for heads & tails
# ---------------------------------------------------------------------------
def split_head_tail(text: str) -> tuple[str, str]:
    """Split *text* into head and tail by the first " – " (strict)."""
    if " – " not in text:
        raise ValueError("Missing head/tail separator ' – ' in: " + text[:120])
    head, _sep, tail = text.partition(" – ")
    return head, tail


def normalize_head(head: str) -> str:
    """Normalise *head* string: trim, flatten newlines, fix naming."""
    cleaned = head.strip().replace("\n", " ")
    return cleaned.replace("Prokuraturę Okręgową", "Prokuratura Okręgowa")

# ---------------------------------------------------------------------------
# Para 4 sub-subpart helpers
# ---------------------------------------------------------------------------

def split_p4_head_tail(text: str) -> tuple[str, str]:
    """
    Split para-4 sub-subpart text on ' dla ' (strict) and normalise the head.

    Returns (head, tail) where head is “Prokuratura Rejonowa …”.
    """
    #if " dla " not in text:
    #    raise ValueError("Missing ' dla ' separator in §4 sub-subpart: " + text[:120])

    #head_raw, tail = text.rsplit(" dla ", 1)
    #matches = list(re.finditer(r"(?<!,) dla ", text))
    #if not matches:
    #    raise ValueError("Missing valid ' dla ' separator in §4 sub-subpart: " + text[:120])

    #idx = matches[-1].start()        # position of the chosen separator
    if " dla części " in text:
        #if "wlotu" in text:
        #    print ('CIESZYŃSKA', text)
        idx = text.find(" dla części ")          # first match
    else:
        # otherwise: last “ dla ” not preceded by “, ”
        matches = list(re.finditer(r"(?<!,) dla ", text))
        if not matches:
            raise ValueError("Missing valid ' dla ' separator in §4 sub-subpart: " + text[:120])
        idx = matches[-1].start()                # last valid match
    head_raw = text[: idx]
    #print ('<'+head_raw+'>')
    tail = text[idx + 5 :]           # len(" dla ") == 5
    if "wlotu" in text:
        print ('CIESZYŃSKA tail', tail)
    #head = head_raw.strip().replace("Prokuraturę Rejonową", "Prokuratura Rejonowa")
    head_clean = re.sub(r"^\d+\)\s*", "", head_raw.strip())
    head = head_clean.replace("Prokuraturę Rejonową", "Prokuratura Rejonowa")
    if "Prokuratura Rejonowa" not in head:  # still strict
        raise ValueError("Bad head after normalisation: " + head_raw[:120])
    return head, tail.strip()


# ---------------------------------------------------------------------------
# Tail-to-list converter for § 4
# ---------------------------------------------------------------------------

_WARSAW_SUB_RX = re.compile(
    r"""części\s+miasta\s+stołecznego\s+Warszawy.*?dzielnic:\s*""",
    flags=re.I | re.S,
)
_CITY_RX   = re.compile(r"""miast[ a]?\s*:?\s*""", flags=re.I)
_ADMIN_CITY_RX = re.compile(
    r"""obszaru\s+administracyjnego\s+miast[ a]?\s*:?\s*""",
    flags=re.I,
)

#_GMINA_RX  = re.compile(r"""gmin[ ay]?\s*:?\s*""", flags=re.I)
#_GMINA_RX = re.compile(
#    r"""gmin(?:a|y)?      # gmina / gminy (optional tail)
#        |gmin             # bare 'gmin' or 'gmin:'
#    """,
#    flags=re.I | re.X,
#)

#_GMINA_RX = re.compile(
#    r"""gmin(?:a|y)?\s*:?\s*""",   # gmin  / gmina / gminy  plus optional colon
#    flags=re.I | re.X,
#)

_GMINA_RX = re.compile(
    r"""gmin(?:a|y)?\s*:?\s*""",   # matches gmin, gmin:, gmina, gminy:
    flags=re.I | re.X,
)

_PART_RX   = re.compile(r"""części\s+""", flags=re.I)

_SEP_RX = re.compile(r",\s+|\s+i\s+|\s+oraz\s+", flags=re.I)

# ---------------------------------------------------------------------------
# Tail-to-list converter for § 4  (handles mid-tail intros)
# ---------------------------------------------------------------------------
bad_WARSAW_INTRO_RX = re.compile(
    r"""części\s+miasta\s+stołecznego\s+Warszawy.*?dzielnic:\s*""",
    flags=re.I | re.S,
)
bad_CITY_INTRO_RX = re.compile(
    r"""(?:miast[ a]?|obszaru\s+administracyjnego\s+miast[ a]?)\s*:?\s*""",
    flags=re.I,
)
bad_GMINA_INTRO_RX = re.compile(r"""\s*gmin(?:a|y)?\s*:?\s*""", flags=re.I)
bad_PART_INTRO_RX = re.compile(r"""części\s+""", flags=re.I)

bad_SEP_RX = re.compile(r",\s+|\s+i\s+|\s+oraz\s+", flags=re.I)

# ---------------------------------------------------------------------------
# Tail-to-list converter for § 4  (supports mid-tail intros, “gmin:…”, etc.)
# ---------------------------------------------------------------------------

_WAW_INTRO_RX  = re.compile(
    r"""części\s+miasta\s+stołecznego\s+Warszawy.*?dzielnic:\s*""",
    flags=re.I | re.S,
)

#_CITY_INTRO_RX = re.compile(
#    r"""(?:miast[ a]?|obszaru\s+administracyjnego\s+miast[ a]?)\s*:?\s*"""
#)

#_CITY_INTRO_RX = re.compile(
#    r"""(?:                         # either …
#          miast[ a]?                #   'miast'  or 'miasta'
#        | obszaru\s+administracyjnego\s+miast[ a]?   # plural form
#        | obszaru\s+administracyjnego\s+miasta       # NEW: singular form
#       )\s*:?\s*""",
#    flags=re.I | re.X,
#)

_CITY_INTRO_RX = re.compile(
    r"""(?:                                     # alternatives
          obszaru\s+administracyjnego\s+miasta   # ← singular form
        | obszaru\s+administracyjnego\s+miast[ a]? # plural form
        | miast[ :]                             # plain 'miast/miasta'
       )\s*:?\s*""",
    flags=re.I | re.X,
)


#_CITY_INTRO_RX = re.compile(
#    r"""(?:                                # alternatives
#          ^miast[ a]?\s*:?\s*$             # ← stand-alone 'miast/miasta'
#        | obszaru\s+administracyjnego\s+miast[ a]?  # longer plural form
#        | obszaru\s+administracyjnego\s+miasta      # longer singular form
#       )""",
#    flags=re.X,
#)

_PLAIN_MIASTA_RX = re.compile(
    r"""(?:
            ^                       |   # start of tail
            ,\s+                    |   # comma-space
            \si\s+                  |   #  i <space>
            \soraz\s+                   #  oraz <space>
        )miasta\s*:?\s+               # the word + opt colon/blanks
    """,
    flags=re.I | re.X,
)


_GMIN_INTRO_RX = re.compile(r"""\s*gmin(?:a|y)?\s*:?\s*""")
_PART_INTRO_RX = re.compile(r"""^części\s+""")

_SEP_RX = re.compile(r",\s+|\s+i\s+|\s+oraz\s+")

def _extract_part_city(phrase: str) -> str:
    """
    From 'części miasta Kraków w …' → return 'Kraków'
    From 'części miasta stołecznego Warszawy …' → 'Warszawy'
    """
    m = re.search(r"""miasta\s+(?:stołecznego\s+)?([A-ZŁŚŻŹĆŃÓ][\w\- ]+)""", phrase, flags=re.I)
    if not m:
        raise ValueError("Cannot extract city name from ‘części …’ phrase: " + phrase[:120])
    return m.group(1).strip()

_WARSZAW_INTRO = re.compile(
    r"""części\s+miasta\s+stołecznego\s+Warszawy.*?dzielnic:\s*""",
    flags=re.I | re.S,
)
_CITY_INTRO  = re.compile(r"""miast[ a]?""", flags=re.I)
_ADMIN_CITY  = re.compile(r"""obszaru\s+administracyjnego\s+miast[ a]?""", flags=re.I)
_GMINA_INTRO = re.compile(r"""gmin(?:a|y)?""", flags=re.I)
_PART_INTRO  = re.compile(r"""części\s+miasta""", flags=re.I)     # descriptive
_SEP_RE      = re.compile(r",\s+|\s+i\s+|\s+oraz\s+", flags=re.I)

_INTRO_RX = re.compile(
    r"""części\s+miasta|
        gmin(?:a|y)?|
        miast[ a]?|
        obszaru\s+administracyjnego\s+miasta              # beginning of “obszaru administracyjnego …”
    """,
    flags=re.X,
)

def _city_from_part(segment: str) -> str:
    """Extract 'Kraków' from 'części miasta Kraków w …' (strict)."""
    m = re.search(
        r"""miasta\s+(?:stołecznego\s+)?([A-ZŁŚŻŹĆŃÓ][\w\- ]+)""",
        segment,
        flags=re.I,
    )
    if not m:
        raise ValueError("Un-parseable descriptive phrase: " + segment[:120])
    return m.group(1).strip()


def parse_p4_tail_list(tail: str) -> list[tuple[str, str]]:
    """
    Returns list[ (name, opis) ]:
        • ordinary locality → (prefixed-name, "")
        • descriptive “części miasta …” → ("m. <city>", full phrase)
    """
    work = tail.lstrip()
    segments: list[str] = []
    debug = False
    if "wlotu" in tail:
        debug=True    
        print ('CIESZYŃSKA tail (inner)', tail)

    # ---- cut the tail into segments by each intro token -----------------
    pos = 0
    for m in _INTRO_RX.finditer(work):
        if m.start() > pos:
            if debug:
                print ('CIESZYŃ segment', work[pos : m.start()].strip())
            segments.append(work[pos : m.start()].strip())
        pos = m.start()
    segments.append(work[pos:].strip())

    tuples: list[tuple[str, str]] = []
    mode: str | None = None        # "city" | "gmina" | "warsaw" | "part"

    for seg in segments:
        if not seg:
            continue

        # 1️⃣ descriptive (“części miasta …”) — keep whole phrase
        #if _PART_INTRO_RX.match(seg):
        #    city = _extract_part_city(seg)
        #    tuples.append((f"m. {city}", seg.rstrip(":")))
        #    mode = "part"
        #    continue
        if _PART_INTRO_RX.match(seg):
            # full descriptive phrase
            phrase = seg.rstrip(":")
            # cut city name before “ w granicach …” (if present)
            city_full = _extract_part_city(seg)
            city = re.split(r"\s+w\s+granicach\b", city_full, 1)[0].rstrip()
            tuples.append((f"m. {city}", phrase))
            mode = "part"
            continue
        
        # 2️⃣ set mode if the segment *starts* with an intro keyword
        first = seg.split(None, 1)[0]  # first word
        if _WAW_INTRO_RX.match(first):
            mode = "warsaw"
            seg = seg[len(first):].lstrip()
        elif _PLAIN_MIASTA_RX.match(seg):
            mode = "city"
            seg = seg[len(first):].lstrip()
        elif _ADMIN_CITY_RX.match(seg):
            mode = "city"
            seg = seg[len(first):].lstrip()
            print ('MATCH ADMIN', '<'+first+'>', '<'+seg+'>')
        elif _CITY_INTRO_RX.match(seg):
            mode = "city"
            seg = seg[len(first):].lstrip()
            print ('MATCH city', '<'+first+'>', '<'+seg+'>')
        elif _GMIN_INTRO_RX.match(first):
            mode = "gmina"
            seg = seg[len(first):].lstrip()

        # 3️⃣ split the remaining chunk only if it isn’t descriptive
        for tok in _SEP_RX.split(seg):
            tok = tok.replace("\n", " ").rstrip(",;.").strip()
            if not tok:
                continue
            if mode == "city":
                tuples.append((f"m. {tok}", ""))
            elif mode == "gmina":
                tuples.append((f"gm. {tok}", ""))
            elif mode in ("warsaw", "part"):
                tuples.append((tok, ""))
            else:
                print ('debug', '<'+first+'>', '<'+seg+'>', _PLAIN_MIASTA_RX.search(seg))
                
                raise ValueError("Locality with no active intro: ", seg, tail)

    return tuples


def parse_p4_tail_listOLD2(tail: str) -> list[tuple[str, str]]:
    """
    Return a list of *(name, opis)* tuples where *opis* is '' unless the
    item is a descriptive *“części miasta …”* segment.
    """
    work = tail.lstrip()
    tuples: list[tuple[str, str]] = []
    mode: str | None = None  # "city" | "gmina" | "warsaw"

    # --- helper to switch mode on a leading intro ------------------------
    def _set_mode(seg: str) -> bool:
        nonlocal mode
        if _WARSZAW_INTRO.match(seg):
            mode = "warsaw"; return True
        if _CITY_INTRO.match(seg) or _ADMIN_CITY.match(seg):
            mode = "city";   return True
        if _GMINA_INTRO.match(seg):
            mode = "gmina";  return True
        return False

    # --------------------------------------------------------------------
    # 1. Split tail into *chunks* by every recognised intro (inclusive)
    # --------------------------------------------------------------------
    chunks: list[str] = []
    pos = 0
    intro_rx = re.compile(
        r"""części\s+miasta|gmin(?:a|y)?|miast[ a]?|obszaru\s+administracyjnego\s+miast[ a]?""",
        flags=re.I,
    )
    for m in intro_rx.finditer(work):
        if m.start() > pos:
            chunks.append(work[pos : m.start()].strip())
        chunks.append(work[m.start() :].split(None, 1)[0])  # the intro word itself
        pos = m.start()
    if pos == 0:                                             # tail starts with intro
        chunks.append(work)
    elif pos < len(work):
        chunks.append(work[pos:].strip())

    # --------------------------------------------------------------------
    # 2. Process each chunk
    # --------------------------------------------------------------------
    for chunk in chunks:
        if not chunk:
            continue

        # --- descriptive phrase -----------------------------------------
        if _PART_INTRO.match(chunk):
            city = _city_from_part(chunk)
            tuples.append((f"m. {city}", chunk.rstrip(":")))
            mode = "part"   # switch to part-mode for any following localities
            continue

        # --- intro keyword sets mode but produces no tuples -------------
        if _set_mode(chunk.split()[0]):   # first word is intro
            # strip the intro word itself from chunk
            chunk = chunk.split(None, 1)[1] if " " in chunk else ""
            if not chunk:
                continue

        # --- ordinary locality list in current mode ---------------------
        for name in _SEP_RE.split(chunk):
            n = name.strip().replace("\n", " ").rstrip(",;")
            if not n:
                continue
            if mode == "city":
                tuples.append((f"m. {n}", ""))
            elif mode == "gmina":
                tuples.append((f"gm. {n}", ""))
            elif mode == "warsaw" or mode == "part":
                tuples.append((n, ""))
            else:
                raise ValueError("Locality with no active intro: " + tail[:120])

    return tuples

# ---------------------------------------------------------------------------
# Intro normaliser for §4 sub-parts
# ---------------------------------------------------------------------------

#_P4_INTRO_RX = re.compile(
#    r"""^\s*w\s+obszarze\s+właściwości\s+
#        Prokuratury\s+Okręgowej\s+
#        (w\s+.+?)\s*:?\s*$""",
#    flags=re.I | re.X,
#)

_P4_INTRO_RX = re.compile(
    r"""^\s*                              # optional leading blanks
        (?:\d+\)\s+)?                     # ← optional '1) ' or '23) '
        w\s+obszarze\s+właściwości\s+      # required phrase
        Prokuratury\s+Okręgowej\s+         # fixed words
        (?P<rest>.+?)                      # capture ‘w X…’ (with or without tag)
        \s*:?\s*$                          # optional colon at the end
    """,
    flags=re.I | re.X,
)



#def normalize_p4_intro(intro: str) -> str:
#    """
#    Convert ‘w obszarze … Prokuratury Okręgowej w X:’ →
#    ‘Prokuratura Okręgowa w X’.
#    Strict: raises ``ValueError`` on mismatch.
#    """
#    m = _P4_INTRO_RX.match(intro)
#    if not m:
#        raise ValueError("Unexpected §4 intro format: " + intro[:120])
#
#    return "Prokuratura Okręgowa " + m.group("rest").strip()

def normalize_p4_intro(text: str) -> str:
    """Return ‘Prokuratura Okręgowa …’ or raise on mismatch (strict)."""
    m = _P4_INTRO_RX.match(text)
    if not m:
        raise ValueError("Unexpected §4 intro format: " + text[:120])
    return "Prokuratura Okręgowa " + m.group("rest").strip()


# ---------------------------------------------------------------------------
# Tail-processing helpers for §3 ▸ subpart 2
# ---------------------------------------------------------------------------
#MAIN_INTROS = (
#    "obejmującą obszar właściwości Prokuratur Rejonowych:",
#    "obejmującą obszar właściwości Prokuratur Rejonowych w:",
#)


_MAIN_INTRO_RX = re.compile(
    r"""^\s*
        obejmującą\s+obszar\s+właściwości\s+
        Prokuratur\s+Rejonowych           # fixed words
        (?:\s+w)?\s* :                    # 0‒1 “ w” then colon
    """,
    flags=re.I | re.X,
)

# quick detector used by the §3 flattening loop
def _is_tail_list(tail: str) -> bool:
    return bool(_MAIN_INTRO_RX.match(tail.lstrip()))



#def _is_tail_list(tail: str) -> bool:
#    return any(tail.lstrip().startswith(intro) for intro in MAIN_INTROS)

_AUX_IW_RE = re.compile(r"\s+i\s*w:")  # " i w:" – keep conjunction
_AUX_W_RE = re.compile(r"\s+w:")       # " w:"  – remove entirely

_UCHYLONA_RE = re.compile(r"^\(uchylona\).\s*$", flags=re.I)

def _strip_main_intro(tail: str) -> str:
    """
    Remove the leading “obejmującą … Prokuratur Rejonowych [w:]” fragment.

    • Works even when stray new-lines or multiple spaces appear inside.
    • Raises ``ValueError`` if the intro is missing or misspelled.
    """
    m = _MAIN_INTRO_RX.match(tail.lstrip()) # m = intro_rx.match(tail)
    if not m:
        raise ValueError("Tail lacks recognised main intro: " + tail[:120])

    return tail[m.end() :]


def _remove_aux_intros(tail: str) -> str:
    tail = _AUX_IW_RE.sub(" i ", tail)   # keep the "i" separator
    return _AUX_W_RE.sub(" ", tail)      # drop bare " w:"


def split_tail_list(tail: str) -> List[str]:
    """
    Clean *tail* and return a list of prosecutor-office names, each
    correctly prefixed according to the following rule-set:

    • If the cleaned name already contains “ w ” or “ we ” **anywhere**, or
      if it *starts* with “w ” / “we ” (case-insensitive), prepend
      “Prokuratura Rejonowa ” (no w/we).

    • Otherwise choose the prefix:
        – “… we ”  if the name starts with capital W followed by a
          consonant (Polish consonants included);
        – “… w ”   in all other cases.
    """
    # 1️⃣  remove main + auxiliary intros, collapse whitespace
    tail = _strip_main_intro(tail)
    tail = _remove_aux_intros(tail.replace("\n", " "))
    tail = re.sub(r"\s{2,}", " ", tail).strip()

    # 2️⃣  split on  “, ”  or  “ i ”
    parts = re.split(r",\s+|\s+i\s+", tail)
    cleaned = [p.rstrip(",;").strip() for p in parts if p.strip()]

    # 3️⃣  consonant set (lower-case) used for the “we” rule
    _CONS = "bcćdfghjklłmnńprqstvwxzźż"

    def _prefix(name: str) -> str:
        s = name.lstrip()
        low = s.lower()

        # already contains or starts with " w " / " we "
        if " w " in low or " we " in low or low.startswith(("w ", "we ")):
            return "Prokuratura Rejonowa "

        # decide between "we" vs "w"
        if s.startswith("W") and len(s) > 1 and s[1].lower() in _CONS:
            return "Prokuratura Rejonowa we "
        return "Prokuratura Rejonowa w "

    return [_prefix(n) + n.lstrip() for n in cleaned]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SubSubPart:
    letter: str
    text: str

    def dump(self, indent: int = 0) -> str:
        return f"{' ' * indent}{self.letter}) {self.text}"

    def __str__(self) -> str:  # pragma: no cover
        return self.dump()


@dataclass
class SubPart:
    number: int
    intro: str
    subsubparts: List[SubSubPart] = field(default_factory=list)
    p4_map: Dict[str, List[tuple[str, str]]] = field(default_factory=dict)   # NEW

    def dump(self, indent: int = 0) -> str:
        pad = " " * indent
        head = f"{pad}{self.number})"
        if self.intro:
            head += f" {self.intro}"
        #lines = [head] + [ssp.dump(indent + 2) for ssp in self.subsubparts]
        lines = [head]

        # § 4 dictionary view
        if self.p4_map:
            d_lines = pprint.pformat(self.p4_map, width=100).splitlines()
            lines += [f"{pad}  {l}" for l in d_lines]
        else:
            lines += [ssp.dump(indent + 2) for ssp in self.subsubparts]
            
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.dump()


@dataclass
class Part:
    number: int
    intro: str
    subparts: List[SubPart] = field(default_factory=list)
    subsubparts: List[SubSubPart] = field(default_factory=list)  # §3 flat
    ss_map: Dict[str, object] = field(default_factory=dict)      # head → tail/list

    def dump(self, indent: int = 0) -> str:
        pad = " " * indent
        lines = [f"{pad}§ {self.number}. {self.intro}"]

        # regular hierarchy
        lines += [sp.dump(indent + 2) for sp in self.subparts]

        # flattened dictionary for §3
        if self.ss_map:
            dict_lines = pprint.pformat(self.ss_map, width=120).splitlines()
            lines += [f"{pad}  {l}" for l in dict_lines]
        else:  # any debug left-overs
            lines += [ssp.dump(indent + 2) for ssp in self.subsubparts]

        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.dump()

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


EXCEL_COLUMNS = ["gmina", "opis", "prokuratura rejonowa", "prokuratura okręgowa"]

def _build_excel_rows(parts: List["Part"]) -> List[dict[str, str]]:
    """Return a list of dictionaries—one per locality—for the Excel file."""
    rows: List[dict[str, str]] = []
    for part in parts:
        if part.number != 4:              # only § 4 carries the mapping
            continue
        for sp in part.subparts:
            okr = sp.intro                # Prokuratura Okręgowa
            for rej, loc_list in sp.p4_map.items():   # head → list[(name, extra)]
                for name, extra in loc_list:
                    rows.append({
                        "gmina": name,
                        "opis":  extra,
                        "prokuratura rejonowa": rej,
                        "prokuratura okręgowa": okr,
                    })
    return rows

def parse_hierarchy(raw: str) -> List[Part]:
    parts: List[Part] = []

    for p_chunk in _segment(PART_RE, raw):
        m_part = re.match(PART_RE, p_chunk)
        if not m_part:
            raise ValueError("Chunk missing PART header")

        part_number = int(re.search(r"\d+", m_part.group(0)).group())
        part_body = p_chunk[m_part.end() :]

        m_first_sub = re.search(SUBPART_RE, part_body, flags=re.M)
        if not m_first_sub:
            raise ValueError(f"§{part_number} lacks subparts")

        part_intro = part_body[: m_first_sub.start()].strip()
        sub_body = part_body[m_first_sub.start() :]
        part_obj = Part(part_number, part_intro)

        for s_chunk in _segment(SUBPART_RE, sub_body):
            m_sub = re.match(SUBPART_RE, s_chunk)
            sub_number = int(re.search(r"\d+", m_sub.group(0)).group())
            sub_body_all = s_chunk[m_sub.end() :]

            # -------- FLATTENED §3 --------
            if part_number == 3:
                m_first_ss = re.search(SUBSUBPART_RE, sub_body_all, flags=re.M)
                if not m_first_ss:
                    raise ValueError("Subpart in §3 missing sub-subparts")
                ss_body = sub_body_all[m_first_ss.start() :]

                for ss_chunk in _segment(SUBSUBPART_RE, ss_body):
                    m_ss = re.match(SUBSUBPART_RE, ss_chunk)
                    letter = re.search(r"[a-z]", m_ss.group(0)).group()
                    ss_text = ss_chunk[m_ss.end() :].strip()

                    head_raw, tail_raw = split_head_tail(ss_text)
                    head = normalize_head(head_raw)

                    if _is_tail_list(tail_raw):
                        tail_value: object = split_tail_list(tail_raw)
                    else:
                        tail_value = tail_raw

                    if head in part_obj.ss_map:
                        raise ValueError(f"Duplicate head '{head}' in §3")
                    part_obj.ss_map[head] = tail_value

                    # keep raw for debugging
                    part_obj.subsubparts.append(SubSubPart(letter, ss_text))

                # Skip building a SubPart for §3
                continue

            # -------- Regular (§≠3) path --------
            m_first_ss = re.search(SUBSUBPART_RE, sub_body_all, flags=re.M)
            if not m_first_ss:
                raise ValueError(f"Subpart {sub_number} in §{part_number} lacks sub-subparts")

            sub_intro_raw = sub_body_all[: m_first_ss.start()].strip()

            # ---------- NEW: normalise §4 intro ---------------------------
            if part_number == 4:
                sub_intro = normalize_p4_intro(sub_intro_raw)
            else:
                sub_intro = sub_intro_raw

            ss_body = sub_body_all[m_first_ss.start() :]
            sub_obj = SubPart(sub_number, sub_intro)

            for ss_chunk in _segment(SUBSUBPART_RE, ss_body):
                m_ss = re.match(SUBSUBPART_RE, ss_chunk)
                letter = re.search(r"[a-z]", m_ss.group(0)).group()
                ss_text = ss_chunk[m_ss.end() :].strip()

                if _UCHYLONA_RE.fullmatch(ss_text):
                    continue

                #sub_obj.subsubparts.append(SubSubPart(letter, ss_text))

                if part_number == 4:
                    head_norm, tail_norm = split_p4_head_tail(ss_text)
                    #ss_text = f"{head_norm} dla {tail_norm}"  # store normalized
                    tail_list = parse_p4_tail_list(tail_norm)
                    if "wlotu" in tail_norm:
                        print ('CIESZYŃSKA tail_list', tail_list)
                    #ss_text = f"{head_norm} dla {tail_list}"
                    # store head for cross-validation later

                    if head_norm in sub_obj.p4_map:
                        raise ValueError(
                            f"Duplicate head '{head_norm}' in §4 subpart {sub_number}"
                        )
                    sub_obj.p4_map[head_norm] = tail_list

                    # keep raw SubSubPart only for debugging (optional)
                    sub_obj.subsubparts.append(SubSubPart(letter, ss_text))
                else:
                    sub_obj.subsubparts.append(SubSubPart(letter, ss_text))
                
            part_obj.subparts.append(sub_obj)

        parts.append(part_obj)

    _validate_links(parts)
    return parts

def _validate_links(parts: List["Part"]) -> None:
    """Ensure every (I,H) pair appears in both §3 and §4."""
    p3_map: Dict[str, set[str]] = {}
    p4_pairs: set[tuple[str, str]] = set()

    # --- gather from §3 -----------------------------------------------------
    for p in parts:
        if p.number == 3:
            for I, val in p.ss_map.items():
                if isinstance(val, list):
                    p3_map[I] = set(val)

    # --- gather & check from §4 --------------------------------------------
    for p in parts:
        if p.number == 4:
            for sp in p.subparts:
                I = sp.intro
                if I not in p3_map:
                    raise ValueError(f"§4 intro '{I}' not present in §3 dictionary")
                for h in sp.p4_map:
                    if h not in p3_map[I]:
                        raise ValueError(
                            f"Head '{h}' under intro '{I}' missing from §3 list"
                        )
                    p4_pairs.add((I, h))
                for ssp in sp.subsubparts:
                    head, _tail = split_p4_head_tail(ssp.text)
                    if head not in p3_map[I]:
                        #print (p3_map)
                        #print (I)
                        #print (p3_map[I])
                        None
                        raise ValueError(f"Head <<{head}>> under intro '{I}' missing from §3 list")
                    p4_pairs.add((I, head))

    # --- ensure §4 covered everything from §3 ------------------------------
    for I, heads in p3_map.items():
        for h in heads:
            if (I, h) not in p4_pairs:
                None
                print ([ p for p in p4_pairs if I==p[0]])
                raise ValueError(f"Missing §4 mapping for head '{h}' under intro '{I}'")

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python hierarchy_parser.py <file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    try:
        structure = parse_hierarchy(text)
        rows = _build_excel_rows(structure)
        if rows:
            df = pd.DataFrame(rows, columns=EXCEL_COLUMNS)
            df.to_excel("prokuratury_mapping.xlsx", index=False)
            print("Excel file written: prokuratury.xlsx")
        for part in structure:
            print(part)
            print()
    except ValueError as exc:
        print("ERROR:", exc)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _cli()
