import unittest
from datetime import datetime, timedelta

from time_slots import generate_time_slots


class TimeSlotTests(unittest.TestCase):
    def test_shahrivar_weekly_range_has_no_extra_partial_week(self):
        # 1405/06/01..19, with the second week crossing a Gregorian month.
        slots = generate_time_slots(
            datetime(2026, 8, 23), datetime(2026, 9, 10), "weekly"
        )
        self.assertEqual(
            [(start.date().isoformat(), end.date().isoformat())
             for start, end in slots],
            [
                ("2026-08-23", "2026-08-29"),  # 1..7 Shahrivar
                ("2026-08-30", "2026-09-05"),  # 8..14 Shahrivar
                ("2026-09-06", "2026-09-10"),  # 15..19 Shahrivar
            ],
        )

    def test_weekly_slots_partition_every_day_exactly_once(self):
        start = datetime(2026, 8, 23)
        for days in (1, 6, 7, 8, 13, 14, 19, 28, 29, 30, 31, 32, 60):
            with self.subTest(days=days):
                end = start + timedelta(days=days - 1)
                slots = generate_time_slots(start, end, "weekly")
                self.assertEqual(len(slots), (days + 6) // 7)
                self.assertEqual(slots[0][0], start)
                self.assertEqual(
                    slots[-1][1], datetime.combine(end.date(), datetime.max.time())
                )
                for i, (slot_start, slot_end) in enumerate(slots):
                    length = slot_end - slot_start + timedelta(microseconds=1)
                    self.assertLessEqual(length, timedelta(days=7))
                    self.assertGreater(length, timedelta(0))
                    if i + 1 < len(slots):
                        self.assertEqual(length, timedelta(days=7))
                        self.assertEqual(
                            slot_end + timedelta(microseconds=1), slots[i + 1][0]
                        )

    def test_other_modes_keep_inclusive_calendar_boundaries(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 4, 2)
        for mode, count, first_end in (
            ("daily", 92, datetime(2026, 1, 1, 23, 59, 59, 999999)),
            ("monthly", 4, datetime(2026, 1, 31, 23, 59, 59, 999999)),
            ("quarterly", 2, datetime(2026, 3, 31, 23, 59, 59, 999999)),
        ):
            with self.subTest(mode=mode):
                slots = generate_time_slots(start, end, mode)
                self.assertEqual(len(slots), count)
                self.assertEqual(slots[0], (start, first_end))
                self.assertEqual(slots[-1][1], datetime(2026, 4, 2, 23, 59, 59, 999999))
        hourly = generate_time_slots(start, start, "hourly")
        self.assertEqual(len(hourly), 24)
        self.assertEqual(hourly[0], (start, datetime(2026, 1, 1, 0, 59, 59, 999999)))
        self.assertEqual(hourly[-1][1], datetime(2026, 1, 1, 23, 59, 59, 999999))


if __name__ == "__main__":
    unittest.main()
