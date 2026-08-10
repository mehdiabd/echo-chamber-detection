import unittest

from community_naming import coerce_allowed_label, get_community_label_color


class CommunityColorTests(unittest.TestCase):
    def test_requested_party_colors_are_stable_for_aliases(self):
        cases = [
            ("اصولگرا", "اصولگرا", "#2563eb"),
            ("پایداری", "پایداری", "#0f172a"),
            ("اصلاح طلب", "اصلاح‌طلب", "#16a34a"),
            ("سلطنت طلب", "سلطنت‌طلب", "#facc15"),
            ("منافق", "منافقین", "#dc2626"),
            ("زن زندگی آزادی", "جریان زن‌زندگی‌آزادی", "#9333ea"),
            ("ری استارتی‌ها", "ری‌استارتی‌ها", "#92400e"),
            ("احمدی نژادی", "احمدی‌نژادی‌ها", "#000000"),
            ("خاکستری", "قشر خاکستری", "#9ca3af"),
            ("عدالت طلب", "عدالت‌خواه", "#ec4899"),
        ]

        for raw_label, canonical_label, color in cases:
            with self.subTest(raw_label=raw_label):
                self.assertEqual(coerce_allowed_label(raw_label), canonical_label)
                self.assertEqual(get_community_label_color(raw_label), color)


if __name__ == "__main__":
    unittest.main()
