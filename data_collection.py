"""X (Twitter) Data Collection Using JSON"""

import json

# Load tweets dataset from res.json
with open("res.json", "r", encoding="utf-8") as f:
    # Skip the "Total hits" line and load the JSON array
    f.readline()  # Skip the first line
    tweets = json.load(f)

# Extract relevant fields for graph construction
messages = []
for tweet in tweets:
    sender = tweet.get("user_id")
    replied_to = tweet.get("entity", {}).get("mention", [])
    for reply in replied_to:
        print(reply)
        messages.append({"sender": sender, "reply_to": reply})

# Print extracted messages
print(messages)
