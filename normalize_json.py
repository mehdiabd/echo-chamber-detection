"""Module providing a function to normalize JSON data."""
import json


def normalize_json(input_file):
    """Function to normalize JSON data by converting Python None to JSON null, etc."""
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Convert Python None to JSON null and False to JSON false
    normalized_data = json.dumps(data, ensure_ascii=False).replace("None", "null") \
        .replace("False", "false")

    with open(input_file, 'w', encoding='utf-8') as file:
        file.write(normalized_data)


if __name__ == "__main__":
    INPUT_FILE = "res.json"  # Input JSON file
    normalize_json(INPUT_FILE)
    print(f"Normalized JSON saved back to {INPUT_FILE}")
