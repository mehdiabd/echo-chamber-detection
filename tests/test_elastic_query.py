import unittest
from datetime import date

from elastic_query import (
    DEFAULT_LOOKBACK_DAYS,
    TOPIC_SEARCH_FIELDS,
    build_topic_clause,
    expand_topic_term,
    parse_topic_terms,
    resolve_date_range,
)


class ElasticQueryTests(unittest.TestCase):
    def test_parses_explicit_or_without_breaking_phrases(self):
        self.assertEqual(
            parse_topic_terms("محیط زیست OR آلودگی هوا OR گرد و خاک"),
            ["محیط زیست", "آلودگی هوا", "گرد و خاک"],
        )

    def test_parses_legacy_delimiters(self):
        self.assertEqual(
            parse_topic_terms("تحریم|sanctions،فشار اقتصادی"),
            ["تحریم", "sanctions", "فشار اقتصادی"],
        )

    def test_expands_persian_spacing_and_vote_variants(self):
        variants = set(expand_topic_term("رای_گیری"))
        self.assertTrue(
            {"رای_گیری", "رای گیری", "رای‌گیری", "رأی گیری", "رأی‌گیری"}
            <= variants
        )

    def test_expands_hashtag_with_and_without_hash(self):
        variants = set(expand_topic_term("#همکاری_ملی"))
        self.assertIn("#همکاری_ملی", variants)
        self.assertIn("همکاری ملی", variants)

    def test_query_uses_verified_fields_and_one_phrase_per_variant(self):
        query = build_topic_clause(parse_topic_terms("معیشت OR گرانی OR تورم"))
        clauses = query["bool"]["should"]
        self.assertGreaterEqual(len(clauses), 3)
        for clause in clauses:
            multi_match = clause["multi_match"]
            self.assertEqual(multi_match["type"], "phrase")
            self.assertEqual(tuple(multi_match["fields"]), TOPIC_SEARCH_FIELDS)
            self.assertNotIn(" OR ", multi_match["query"])

    def test_default_range_is_fourteen_days_inclusive(self):
        self.assertEqual(DEFAULT_LOOKBACK_DAYS, 14)
        start, end = resolve_date_range(
            start_value=None,
            end_value=None,
            today=date(2026, 7, 27),
        )
        self.assertEqual(start, date(2026, 7, 14))
        self.assertEqual(end, date(2026, 7, 27))
        self.assertEqual((end - start).days + 1, 14)

    def test_explicit_dates_override_default_window(self):
        start, end = resolve_date_range(
            start_value="2026-07-01",
            end_value="2026-07-10",
        )
        self.assertEqual(start, date(2026, 7, 1))
        self.assertEqual(end, date(2026, 7, 10))

    def test_rejects_reversed_date_range(self):
        with self.assertRaises(ValueError):
            resolve_date_range("2026-07-20", "2026-07-10")


if __name__ == "__main__":
    unittest.main()
