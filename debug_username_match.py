"""Debug username matching"""
import json

print("=== Checking username matching ===\n")

# Load res.json
print("Loading res.json...")
with open("res.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} items\n")

# Get unique usernames from res.json
usernames_in_file = set()
for item in data:
    username = (item.get("user_name") or 
               item.get("sender") or 
               item.get("username") or
               item.get("user_id"))
    if username:
        usernames_in_file.add(username)

print(f"Found {len(usernames_in_file)} unique usernames in res.json")
print(f"Sample usernames: {list(usernames_in_file)[:10]}\n")

# Test with some community members
test_members = ["PahlaviReza", "netanyahu", "Dr_Amir_Hamidi", "ChaawTimes", "last_cry7"]

print("Testing community members:")
for member in test_members:
    if member in usernames_in_file:
        print(f"  ✓ {member} - FOUND")
        # Count texts
        count = sum(1 for item in data if (
            item.get("user_name") == member or
            item.get("sender") == member or
            item.get("username") == member
        ))
        print(f"    Has {count} texts")
    else:
        print(f"  ✗ {member} - NOT FOUND")

print(f"\n=== Checking case sensitivity ===")
# Check if names exist with different case
for member in test_members:
    matches = [u for u in usernames_in_file if u.lower() == member.lower()]
    if matches:
        print(f"  {member} → Found as: {matches}")
