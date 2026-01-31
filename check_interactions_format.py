"""Check interactions.json format"""
import json

print("=== Checking interactions.json ===\n")

with open("interactions.json", "r", encoding="utf-8") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        
        try:
            data = json.loads(line.strip())
            print(f"Line {i+1}:")
            print(f"  Keys: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, str):
                    print(f"  {key}: {value[:100]}")
                else:
                    print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}\n")
