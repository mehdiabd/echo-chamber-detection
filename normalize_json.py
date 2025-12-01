"""Module providing a function to normalize JSON data."""
import json
import ast


def normalize_json(input_file):
    """Normalize the JSON data in the file."""
    valid_json_list = []
    total_hits = None

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Extract "Total hits" line
            if line.startswith("Total hits:"):
                total_hits = line
                continue
            # Skip empty lines
            if not line:
                continue
            try:
                # Parse line as JSON
                python_dict = json.loads(line)
                # print("Parsed line:", python_dict)
                if not all(k in python_dict for k in ["user_name", "user_title", "political_label"]):
                    # print(f"❌ Skipped line (missing required keys): {python_dict}")
                    continue
                else:
                    # print(f"✅ Kept line: {python_dict}")
                    valid_json_list.append(python_dict)
            except json.JSONDecodeError as e:
                print(e)
                continue

    # Write the "Total hits" line and valid JSON array back to the file
    with open(input_file, 'w', encoding='utf-8') as file:
        if total_hits:
            file.write(total_hits + "\n")  # Write the "Total hits" line
        json.dump(valid_json_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    INPUT_FILE = "res.json"  # Input JSON file
    normalize_json(INPUT_FILE)
    print(f"Normalized JSON saved back to {INPUT_FILE}")
