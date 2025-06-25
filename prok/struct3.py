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
        w\s+obszarze\s+właściwości\s+      # required phrase
        Prokuratury\s+Okręgowej\s+         # fixed words
        (?P<rest>.+?)                      # capture ‘w X…’ (with or without tag)
        \s*:?\s*$                          # optional colon at the end
    """,
    flags=re.I | re.X,
)



def normalize_p4_intro(intro: str) -> str:
    """
    Convert ‘w obszarze … Prokuratury Okręgowej w X:’ →
    ‘Prokuratura Okręgowa w X’.
    Strict: raises ``ValueError`` on mismatch.
    """
    m = _P4_INTRO_RX.match(intro)
    if not m:
        raise ValueError("Unexpected §4 intro format: " + intro[:120])

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



def _strip_main_intro(tail: str) -> str:
    """
    Remove the leading “obejmującą … Prokuratur Rejonowych [w:]” fragment.

    • Works even when stray new-lines or multiple spaces appear inside.
    • Raises ``ValueError`` if the intro is missing or misspelled.
    """
    #intro_rx = re.compile(
    #    r"""^\s*                                   # any leading blanks
    #        obejmującą\s+obszar\s+właściwości\s+
    #        Prokuratur\s+Rejonowych                # core phrase
    #        (?:\s+w)?\s*:                          # optional ' w' + colon
    #        \s*                                    # trailing blanks/new-lines
    #    """,
    #    flags=re.I | re.X,
    #)

    m = _MAIN_INTRO_RX.match(tail.lstrip()) # m = intro_rx.match(tail)
    if not m:
        raise ValueError("Tail lacks recognised main intro: " + tail[:120])

    return tail[m.end() :]




#def _strip_main_introOLD(tail: str) -> str:
#    work = tail.lstrip()
#    for intro in MAIN_INTROS:
#        if work.startswith(intro):
#            return work[len(intro) :]
#    raise ValueError("Tail lacks recognised main intro: " + tail[:120])


def _remove_aux_intros(tail: str) -> str:
    tail = _AUX_IW_RE.sub(" i ", tail)   # keep the "i" separator
    return _AUX_W_RE.sub(" ", tail)      # drop bare " w:"


def split_tail_list(tail: str) -> List[str]:
    """Convert *tail* into a list of ‘Prokuratura Rejonowa …’ strings."""
    # 1️⃣ remove the main intro and auxiliary fragments
    tail = _strip_main_intro(tail)
    tail = _remove_aux_intros(tail.replace("\n", " "))

    # 2️⃣ collapse duplicate whitespace
    tail = re.sub(r"\s{2,}", " ", tail).strip()

    # 3️⃣ split on “, ” or “ i ”
    pieces = re.split(r",\s+|\s+i\s+", tail)
    parts = [p.rstrip(",;").strip() for p in pieces if p.strip()]

    # 4️⃣ prepend the required phrase
    def _pref(s: str) -> str:
        return (
            "Prokuratura Rejonowa " if " w " in s else "Prokuratura Rejonowa w "
        ) + s

    return [_pref(p) for p in parts]

def split_tail_listOLD(tail: str) -> List[str]:
    """Return *tail* as a list of prosecutor-office strings.

    Steps
    -----
    1. Strip the main intro (“obejmującą obszar …”).
    2. Remove every auxiliary intro (“ w:” and “ i w:”).
    3. Replace newlines with spaces and collapse double spaces.
    4. Split the cleaned text on “, ” or “ i ”.
    5. Prepend:
         • “Prokuratura Rejonowa ”  if the piece already contains “ w ”
         • “Prokuratura Rejonowa w ” otherwise.
    """
    # --- 1: strict removal of the main intro ------------------------------
    tail = _strip_main_intro(tail)

    # --- 2 & 3: auxiliary intro removal + whitespace normalisation --------
    tail = _remove_aux_intros(tail.replace("\n", " "))
    tail = re.sub(r"\s{2,}", " ", tail).strip()

    # --- 4: split into individual locality strings -----------------------
    pieces = re.split(r",\s+|\s+i\s+", tail)
    cleaned = [p.rstrip(",").strip() for p in pieces if p.strip()]

    # --- 5: prepend the correct phrase -----------------------------------
    def _with_prefix(s: str) -> str:
        return ("Prokuratura Rejonowa " if " w " in s else "Prokuratura Rejonowa w ") + s

    return [_with_prefix(p) for p in cleaned]

def split_tail_listBAD(tail: str) -> List[str]:
    """Return *tail* as list after strict clean-up (para 3 ▸ subpart 2)."""
    tail = _strip_main_intro(tail)

    tail = _remove_aux_intros(tail.replace("\n", " "))
    # collapse double spaces produced by the replacements
    tail = re.sub(r"\s{2,}", " ", tail)
    items = re.split(r",\s+|\s+i\s+", tail)
    # strip and drop any empty / trailing-comma artefacts
    return [i.rstrip(",").strip() for i in items if i.strip()]
    #tail = _remove_aux_intros(tail)
    #items = re.split(r",\s+|\s+i\s+", tail)
    #return [i.strip() for i in items if i.strip()]

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

    def dump(self, indent: int = 0) -> str:
        pad = " " * indent
        head = f"{pad}{self.number})"
        if self.intro:
            head += f" {self.intro}"
        lines = [head] + [ssp.dump(indent + 2) for ssp in self.subsubparts]
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

            sub_intro = sub_body_all[: m_first_ss.start()].strip()

            # ---------- NEW: normalise §4 intro ---------------------------
            if part_number == 4:
                sub_intro = normalize_p4_intro(sub_intro)

            ss_body = sub_body_all[m_first_ss.start() :]
            sub_obj = SubPart(sub_number, sub_intro)

            for ss_chunk in _segment(SUBSUBPART_RE, ss_body):
                m_ss = re.match(SUBSUBPART_RE, ss_chunk)
                letter = re.search(r"[a-z]", m_ss.group(0)).group()
                ss_text = ss_chunk[m_ss.end() :].strip()
                sub_obj.subsubparts.append(SubSubPart(letter, ss_text))

            part_obj.subparts.append(sub_obj)

        parts.append(part_obj)

    return parts

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
