"""Check res.json format"""
import json

# Check first 5 lines of res.json
print("=== Checking res.json format ===\n")

with open("res.json", "r", encoding="utf-8") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        
        print(f"Line {i+1}:")
        try:
            data = json.loads(line)
            print(f"  Keys: {list(data.keys())}")
            
            # Check for username fields
            username_fields = ['user_name', 'sender', 'username', 'user_id']
            for field in username_fields:
                if field in data:
                    print(f"  {field}: {data[field]}")
            
            # Check for text fields
            text_fields = ['normalized_text', 'text', 'content']
            for field in text_fields:
                if field in data and data[field]:
                    print(f"  {field}: {data[field][:100]}...")
            
            print()
        except json.JSONDecodeError as e:
            print(f"  ERROR: {e}\n")
            print(f"  Raw line: {line[:200]}\n")
