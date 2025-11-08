"""Helper module for saving community details."""

import json
from datetime import datetime


def save_community_details(community_info, slot_start=None, slot_end=None):
    """Save community details to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add time slot info if provided
    if slot_start and slot_end:
        time_info = {
            "slot_start": slot_start.strftime("%Y-%m-%d"),
            "slot_end": slot_end.strftime("%Y-%m-%d")
        }
    else:
        time_info = {}
    
    # Format data for saving
    data = {
        "timestamp": timestamp,
        "time_slot": time_info,
        "communities": community_info
    }
    
    # Save to file
    filename = f"community_details_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[saved] Community details saved to {filename}")