#!/usr/bin/env python3
"""
Force regenerate the last dashboard page with fresh community names
"""
import os
import glob
import json
from datetime import datetime

def force_regenerate_last_dashboard():
    """Find and delete the last dashboard files to force regeneration"""
    
    # Find all dashboard files
    dashboards = sorted([f for f in glob.glob("dashboard_*.html") if "_to_" in f])
    
    if not dashboards:
        print("No dashboard files found.")
        return
    
    last_dashboard = dashboards[-1]
    print(f"Found last dashboard: {last_dashboard}")
    
    # Find related files
    base_name = last_dashboard.replace(".html", "")
    related_files = [
        last_dashboard,
        f"{base_name}_legend.json",
        last_dashboard.replace("dashboard_", "louvain_graph_"),
        last_dashboard.replace("dashboard_", "hybrid_graph_")
    ]
    
    # Delete all related files
    deleted = []
    for file in related_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                deleted.append(file)
                print(f"✓ Deleted: {file}")
            except Exception as e:
                print(f"✗ Failed to delete {file}: {e}")
    
    if deleted:
        print(f"\n✓ Successfully deleted {len(deleted)} files")
        print("\nNow run your community_detection.py again to regenerate with fresh names!")
    else:
        print("\n✗ No files were deleted")

if __name__ == "__main__":
    force_regenerate_last_dashboard()
