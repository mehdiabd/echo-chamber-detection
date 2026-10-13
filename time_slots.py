"""Shared calendar boundaries for pipeline processing and dashboard lookup."""

from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta


def generate_time_slots(start_date, end_date, slot_type):
    """Partition inclusive dates, anchoring every slot to the requested start.

    Weekly slots always contain seven days, except for a shorter final slot.
    End timestamps are inclusive so adjacent slots cannot share a message.
    """
    steps = {
        "hourly": timedelta(hours=1),
        "daily": timedelta(days=1),
        "weekly": timedelta(days=7),
        "monthly": relativedelta(months=1),
        "quarterly": relativedelta(months=3),
    }
    if slot_type not in steps:
        raise ValueError(f"Unsupported slot type: {slot_type}")
    current = datetime.combine(start_date.date(), datetime.min.time())
    range_end = datetime.combine(end_date.date(), datetime.max.time())
    slots = []
    while current <= range_end:
        next_slot = current + steps[slot_type]
        slot_end = min(next_slot - timedelta(microseconds=1), range_end)
        slots.append((current, slot_end))
        current = next_slot
    return slots
