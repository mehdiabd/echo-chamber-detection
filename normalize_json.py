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
                # Convert Python-style dict to JSON
                python_dict = ast.literal_eval(line)
                valid_json_list.append(python_dict)
            except (ValueError, SyntaxError):
                print(f"Skipping invalid line: {line}")

    # Write the "Total hits" line and valid JSON array back to the file
    with open(input_file, 'w', encoding='utf-8') as file:
        if total_hits:
            file.write(total_hits + "\n")  # Write the "Total hits" line
        json.dump(valid_json_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    INPUT_FILE = "res.json"  # Input JSON file
    normalize_json(INPUT_FILE)
    print(f"Normalized JSON saved back to {INPUT_FILE}")
