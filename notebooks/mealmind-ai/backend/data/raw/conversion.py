import csv
import json

def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Reads a CSV file and saves its content into a JSON file.
    """
    data = []
    
    # 1. Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Use DictReader to automatically map headers to dictionary keys
        csv_reader = csv.DictReader(csv_file)
        
        # 2. Append each row to the data list
        for row in csv_reader:
            data.append(row)

    # 3. Write data to the JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        # indent=4 makes the JSON file human-readable
        json.dump(data, json_file, indent=4)

# Usage Example:
# Replace 'input.csv' and 'output.json' with your actual file names
convert_csv_to_json('full_dataset.csv', 'full_dataset.json')