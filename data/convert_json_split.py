import json
import re

# Load the original JSON file
with open('data/rag_optimized_5000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to split content into question and answer
def split_content(content):
    question_match = re.search(r'Question:\s*(.*?)\n', content, re.DOTALL)
    answer_match = re.search(r'Answer:\s*(.*)', content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ''
    answer = answer_match.group(1).strip() if answer_match else ''
    return question, answer

# Transform data
new_data = []
for entry in data:
    question, answer = split_content(entry.get('content', ''))
    metadata = entry.get('metadata', {})
    new_entry = {
        'id': entry.get('id'),
        'title': entry.get('title'),
        'question': question,
        'answer': answer,
        'category': entry.get('category'),
        'type': metadata.get('type', ''),
        'difficulty': metadata.get('difficulty', ''),
        'audience': metadata.get('audience', '')
    }
    new_data.append(new_entry)

# Save the new JSON file
with open('data/rag_optimized_5000_split.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
