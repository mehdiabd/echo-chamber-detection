import tempfile
import unittest
from pathlib import Path

from party_change_api import PartyChangeStore, validate_events


def event(event_id="event-1", node_id="alice"):
    return {
        "event_id": event_id,
        "node_id": node_id,
        "from_party": "party-a",
        "to_party": "party-b",
        "previous_slot": "2026-01-01",
        "current_slot": "2026-01-02",
        "slot_mode": "daily",
        "topic": "topic",
        "detected_at": "2026-01-02T12:00:00Z",
    }


class PartyChangeStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PartyChangeStore(
            Path(self.temp_dir.name) / "events.sqlite3",
            retention_days=0,
            max_events=100,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_upsert_is_idempotent_and_queryable(self):
        self.assertEqual(self.store.upsert_many([event()]), 1)
        updated = event()
        updated["to_party"] = "party-c"
        self.assertEqual(self.store.upsert_many([updated]), 1)

        total, events = self.store.list({"node_id": "alice"}, 100, 0)
        self.assertEqual(total, 1)
        self.assertEqual(events[0]["to_party"], "party-c")
        self.assertEqual(self.store.get("event-1")["node_id"], "alice")

    def test_max_event_rotation(self):
        store = PartyChangeStore(
            Path(self.temp_dir.name) / "rotated.sqlite3",
            retention_days=0,
            max_events=1,
        )
        first = event("event-1")
        second = event("event-2", "bob")
        second["detected_at"] = "2026-01-03T12:00:00Z"
        store.upsert_many([first, second])

        total, events = store.list({}, 100, 0)
        self.assertEqual(total, 1)
        self.assertEqual(events[0]["event_id"], "event-2")

    def test_validation_rejects_non_transition(self):
        invalid = event()
        invalid["to_party"] = invalid["from_party"]
        with self.assertRaises(ValueError):
            validate_events({"events": [invalid]})


if __name__ == "__main__":
    unittest.main()
