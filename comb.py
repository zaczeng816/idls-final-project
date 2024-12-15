import json
from pathlib import Path

# Path to the directory containing the JSON files
data_dir = Path("data/result")
output_file = "flattened_gsm8k_questions.json"

# Initialize an empty list to hold all the labeled questions and their solutions
all_entries = []

def sort_key(x):
    try:
        start_idx = int(x.stem.split("_")[-2])
        batch_number = int(x.stem.split("_")[-1])
        return start_idx * 1000000 + batch_number
    except ValueError:
        return 0

def load_questions(file_path: str):
   """Load questions from start_idx to end_idx from the GSM8K JSONL file."""
   questions = []
   with open(file_path, 'r') as f:
       for i, line in enumerate(f):
            questions.append(json.loads(line.strip())['question'])
   return questions


questions_original = load_questions("data/train.jsonl")
q_to_idx = {q: i for i, q in enumerate(questions_original)}



paths = list(data_dir.glob("gsm8k_solutions_batch_*.json"))
paths.sort(key=sort_key)

# Loop through all batch files
for batch_file in sorted(paths):
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
        
        # Track global question index
        question_tracker = {}  # Tracks question_text -> question_number mapping
        
        for pair in batch_data["pairs"]:
            question_text = pair["question"]
            question_number = q_to_idx[question_text]
            
            batch_number = question_number//20
            # Add the entry for this solution pair
            all_entries.append({
                "question_number": question_number,
                "batch_number": batch_number,
                "question": question_text,
                "reasoning": pair["reasoning"],
                "predict_answer": pair["predict_answer"],
                "correct_answer": pair["correct_answer"],
                "approach_info": pair["approach_info"]
            })

all_entries.sort(key=lambda x: x["question_number"])

# Write the flattened list to a single JSON file
with open(output_file, 'w') as f:
    json.dump(all_entries, f, indent=2)

print(f"Flattened {len(all_entries)} question-solution pairs into {output_file}.")
