import json
import csv

# Load the split JSON file
with open('data/rag_optimized_5000_split.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define CSV columns
fieldnames = ['id', 'title', 'question', 'answer', 'category', 'type', 'difficulty', 'audience']

# Write to CSV
with open('data/rag_optimized_5000_split.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)
