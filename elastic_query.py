"""Pure helpers for building topic queries and resolving their date range."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Iterable, List, Sequence, Tuple


DEFAULT_LOOKBACK_DAYS = 14

TOPIC_SEARCH_FIELDS = (
    "normalized_text^3",
    "text^2",
    "entity.hashtag^4",
)

_TOPIC_SEPARATOR_RE = re.compile(
    r"(?:[,،;\n|]+|\s+\bOR\b\s+|\s+یا\s+)",
    flags=re.IGNORECASE,
)
_TERM_JOINER_RE = re.compile(r"[\s_\u200c]+")
_PERSIAN_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
    }
)


def _strip_matching_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def parse_topic_terms(raw_query: str | None) -> List[str]:
    """Split the configured OR-list without breaking multi-word phrases."""
    if not raw_query:
        return []
    return [
        cleaned
        for item in _TOPIC_SEPARATOR_RE.split(raw_query)
        if (cleaned := _strip_matching_quotes(item))
    ]


def expand_topic_term(term: str) -> List[str]:
    """Return controlled Persian spacing and character variants for one term."""
    cleaned = _strip_matching_quotes(term)
    if not cleaned:
        return []

    has_hash = cleaned.startswith("#")
    lexical = cleaned[1:] if has_hash else cleaned
    lexical = lexical.strip()
    if not lexical:
        return []

    raw_spaced = _TERM_JOINER_RE.sub(" ", lexical).strip()
    normalized_spaced = raw_spaced.translate(_PERSIAN_TRANSLATION)
    spaced_forms = {raw_spaced, normalized_spaced}

    # رای and رأی both occur frequently and are not unified by the index analyzer.
    for value in tuple(spaced_forms):
        tokens = value.split()
        if "رای" in tokens:
            spaced_forms.add(" ".join("رأی" if token == "رای" else token for token in tokens))
        if "رأی" in tokens:
            spaced_forms.add(" ".join("رای" if token == "رأی" else token for token in tokens))

    variants = set()
    for spaced in spaced_forms:
        if not spaced:
            continue
        variants.add(spaced)
        variants.add(spaced.replace(" ", "_"))
        variants.add(spaced.replace(" ", "\u200c"))

    # Hashtag values differ across index versions: some retain '#', some do not.
    if has_hash:
        variants.update(f"#{value}" for value in tuple(variants))

    return sorted(variants)


def build_topic_clause(
    topic_terms: Iterable[str],
    fields: Sequence[str] = TOPIC_SEARCH_FIELDS,
) -> dict:
    """Build an OR of exact phrase clauses over the verified text fields."""
    should = []
    seen = set()
    for term in topic_terms:
        for variant in expand_topic_term(term):
            key = (variant, tuple(fields))
            if key in seen:
                continue
            seen.add(key)
            should.append(
                {
                    "multi_match": {
                        "query": variant,
                        "fields": list(fields),
                        "type": "phrase",
                        "zero_terms_query": "none",
                    }
                }
            )

    if not should:
        return {"match_none": {}}
    return {"bool": {"should": should, "minimum_should_match": 1}}


def parse_date_arg(value: str | None, default: date) -> date:
    if not value:
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_date_range(
    start_value: str | None,
    end_value: str | None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    today: date | None = None,
) -> Tuple[date, date]:
    """Resolve an inclusive range, defaulting to the latest 14 calendar days."""
    resolved_end = parse_date_arg(end_value, today or date.today())
    days = max(1, lookback_days)
    resolved_start = parse_date_arg(
        start_value,
        resolved_end - timedelta(days=days - 1),
    )
    if resolved_start > resolved_end:
        raise ValueError(
            f"start date {resolved_start.isoformat()} is after "
            f"end date {resolved_end.isoformat()}"
        )
    return resolved_start, resolved_end
