"""
Abbreviation expansion preprocessor.

Loads an abbreviation dictionary CSV with columns (PubMed_ID, abbreviation,
full_name) and rewrites unambiguous abbreviations in a user question into the
form ``full name (ABBR)`` before retrieval. Ambiguous abbreviations (multiple
distinct full names across the corpus) are skipped so the original question is
preserved.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Pattern, Tuple


_WS_RE = re.compile(r"\s+")


def _normalize_full_name(name: str) -> str:
    """Normalize a full-name string for ambiguity grouping (lowercase, collapsed whitespace)."""
    return _WS_RE.sub(" ", (name or "").strip().lower())


def load_abbreviation_dict(csv_path: Path | str) -> Dict[str, str]:
    """
    Load the abbreviation CSV and return ``{abbreviation: canonical_full_name}``
    containing only unambiguous abbreviations.

    An abbreviation is kept only when all its rows agree on a single normalized
    full name (case-insensitive, whitespace-collapsed). The value is the most
    frequent original-casing surface form for that abbreviation.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    surface_counts: Dict[str, Counter] = defaultdict(Counter)
    normalized_groups: Dict[str, set] = defaultdict(set)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = (row.get("abbreviation") or "").strip()
            full = (row.get("full_name") or "").strip()
            if not abbr or not full:
                continue
            norm = _normalize_full_name(full)
            if not norm:
                continue
            normalized_groups[abbr].add(norm)
            surface_counts[abbr][full] += 1

    abbr_dict: Dict[str, str] = {}
    for abbr, norms in normalized_groups.items():
        if len(norms) != 1:
            continue
        counts = surface_counts.get(abbr)
        if not counts:
            continue
        canonical = counts.most_common(1)[0][0]
        abbr_dict[abbr] = canonical
    return abbr_dict


def build_abbreviation_regex(abbr_dict: Dict[str, str]) -> Pattern[str] | None:
    """
    Compile a single regex that matches any abbreviation key as a whole token.

    Keys are sorted by length descending so that longer abbreviations win over
    shorter prefixes. A non-alphanumeric boundary (lookaround) is used so that
    keys containing non-word characters like ``(+)-ABA`` or ``MLNs`` still match
    correctly.
    """
    keys = [k for k in abbr_dict.keys() if k]
    if not keys:
        return None
    keys.sort(key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in keys)
    pattern = r"(?<![A-Za-z0-9])(?:" + alternation + r")(?![A-Za-z0-9])"
    return re.compile(pattern)


def _context_already_has_full_name(
    text: str, full_name: str, match_start: int, match_end: int, window: int = 80
) -> bool:
    """
    Return True if ``full_name`` already appears near the match (case-insensitive),
    so we should skip expanding to avoid duplicates like ``macrolactins (macrolactins (MLNs))``.
    """
    full_low = full_name.strip().lower()
    if not full_low:
        return False
    left = max(0, match_start - window)
    right = min(len(text), match_end + window)
    window_text = text[left:right].lower()
    return full_low in window_text


def expand_abbreviations(
    text: str,
    abbr_dict: Dict[str, str],
    regex: Pattern[str] | None,
) -> Tuple[str, List[dict]]:
    """
    Expand unambiguous abbreviations in ``text`` to ``full name (ABBR)``.

    Returns ``(new_text, replacements)`` where ``replacements`` is a list of
    ``{"abbr": ..., "full_name": ..., "span": (start, end)}`` for each
    replacement actually applied (skipped candidates are not listed). If no
    replacements are applied the original ``text`` is returned unchanged.
    """
    if not text or not abbr_dict or regex is None:
        return text, []

    replacements: List[dict] = []
    out_parts: List[str] = []
    cursor = 0
    last_emitted_full_name: str | None = None

    for m in regex.finditer(text):
        abbr = m.group(0)
        full_name = abbr_dict.get(abbr)
        if not full_name:
            continue
        start, end = m.span()
        if _context_already_has_full_name(text, full_name, start, end):
            continue
        if (
            last_emitted_full_name is not None
            and last_emitted_full_name.lower() == full_name.lower()
        ):
            continue
        out_parts.append(text[cursor:start])
        out_parts.append(f"{full_name} ({abbr})")
        cursor = end
        replacements.append(
            {
                "abbr": abbr,
                "full_name": full_name,
                "span": (start, end),
            }
        )
        last_emitted_full_name = full_name

    if not replacements:
        return text, []

    out_parts.append(text[cursor:])
    return "".join(out_parts), replacements
