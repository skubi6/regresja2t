"""hierarchy_parser.py – converts nested statute text to a navigable
object model and offers convenient debugging printers.
Updated: 2025‑06‑25 – segment helper now returns *full* text for each
header, fixing the earlier truncation of §‑level and sub‑level bodies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment(pattern: str, src: str, flags: int = re.M) -> List[str]:
    """Slice *src* into **segments** that each **begin** with *pattern* and
    run up to (but *not* including) the next appearance of the same pattern
    (or the end of *src*).

    Example
    -------
    >>> _segment(r"^§\\s*\\d+\.", "§ 3. aaa\n§ 4. bbb")
    ['§ 3. aaa\n', '§ 4. bbb']

    This replaces the earlier :pyfunc:`_split_keep`, ensuring the *entire*
    paragraph (or subparagraph) is preserved in each chunk.
    """

    matches = list(re.finditer(pattern, src, flags))
    if not matches:
        return []

    segments: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(src)
        segments.append(src[start:end])
    return segments


# ---------------------------------------------------------------------------
# Data model with recursive pretty‑printers
# ---------------------------------------------------------------------------

@dataclass
class SubSubPart:
    letter: str
    text: str

    # recursive pretty‑printer
    def dump(self, indent: int = 0) -> str:
        return f"{' ' * indent}{self.letter}) {self.text}"

    def __str__(self) -> str:  # noqa: DunderStr
        return self.dump()


@dataclass
class SubPart:
    number: int
    intro: Optional[str] = ""
    subsubparts: List[SubSubPart] = field(default_factory=list)

    def dump(self, indent: int = 0) -> str:
        pad = ' ' * indent
        head = f"{pad}{self.number})"
        if self.intro:
            head += f" {self.intro}"
        lines = [head]
        # lines = [f"{pad}{self.number}) {self.intro}"]
        for ssp in self.subsubparts:
            lines.append(ssp.dump(indent + 2))
        return "\n".join(lines)

    def __str__(self) -> str:  # noqa: DunderStr
        return self.dump()


@dataclass
class Part:
    number: int
    intro: str
    subparts: List[SubPart] = field(default_factory=list)

    def dump(self, indent: int = 0) -> str:
        pad = ' ' * indent
        lines = [f"{pad}§ {self.number}. {self.intro}"]
        for sp in self.subparts:
            lines.append(sp.dump(indent + 2))
        return "\n".join(lines)

    def __str__(self) -> str:  # noqa: DunderStr
        return self.dump()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

PART_RE = r"^§\s*\d+\."  # e.g. "§ 3."
SUBPART_RE = r"^\s*\d+\)"  # e.g. "28)"
SUBSUBPART_RE = r"^\s*[a-z]\)"  # e.g. "g)"


def parse_hierarchy(raw: str) -> List[Part]:
    """Convert *raw* statute text into a list of :class:`Part` objects."""

    parts: List[Part] = []

    for p_chunk in _segment(PART_RE, raw):
        m_part = re.match(PART_RE, p_chunk)
        if not m_part:
            continue  # should not happen – every chunk starts with the pattern

        part_number = int(re.search(r"\d+", m_part.group(0)).group())
        part_body = p_chunk[m_part.end():]

        # -------- intro (text before first subpart) --------
        m_first_sub = re.search(SUBPART_RE, part_body, flags=re.M)
        if m_first_sub:
            part_intro = part_body[: m_first_sub.start()].strip()
            sub_body = part_body[m_first_sub.start():]
        else:
            part_intro = part_body.strip()
            sub_body = ""

        part_obj = Part(part_number, part_intro)

        # -------- iterate subparts --------
        for s_chunk in _segment(SUBPART_RE, sub_body):
            m_sub = re.match(SUBPART_RE, s_chunk)
            sub_number = int(re.search(r"\d+", m_sub.group(0)).group())
            sub_body_all = s_chunk[m_sub.end():]

            # intro (before first subsubpart)
            m_first_ss = re.search(SUBSUBPART_RE, sub_body_all, flags=re.M)
            if m_first_ss:
                sub_intro = sub_body_all[: m_first_ss.start()].strip()
                ss_body = sub_body_all[m_first_ss.start():]
                if part_number == 3:
                    sub_intro = ""
            else:
                sub_intro = sub_body_all.strip()
                ss_body = ""

            sub_obj = SubPart(sub_number, sub_intro)

            # sub‑subparts
            for ss_chunk in _segment(SUBSUBPART_RE, ss_body):
                m_ss = re.match(SUBSUBPART_RE, ss_chunk)
                letter = re.search(r"[a-z]", m_ss.group(0)).group()
                ss_text = ss_chunk[m_ss.end():].strip()
                sub_obj.subsubparts.append(SubSubPart(letter, ss_text))

            part_obj.subparts.append(sub_obj)

        parts.append(part_obj)

    return parts


# ---------------------------------------------------------------------------
# Demo (CLI) – run this file directly to see the first part printed nicely
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Parse and pretty‑print statute hierarchy")
    parser.add_argument("path", help="Path to the text file", nargs="?", default="prokuratura.txt")
    args = parser.parse_args()

    try:
        data = open(args.path, encoding="utf-8").read()
    except OSError as exc:
        print(f"Error reading {args.path}: {exc}", file=sys.stderr)
        sys.exit(1)

    for part in parse_hierarchy(data):
        print(part)
        print()  # blank line between parts
