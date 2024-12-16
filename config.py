import os
import torch
import json
from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass
class Config:
    # Model configs
    student1_model: str = "google/gemma-2-2b"
    student2_model: str = "google/gemma-2-2b" 
    token: str = "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"
    
    # Data configs
    train_file: str = "train.json"
    val_file: str = "valid.json"
    max_seq_length: int = 512
    consistency_threshold: float = 0.7
    
    # Training configs
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Paths
    output_dir_student1: str = "./student1_model"
    output_dir_student2: str = "./student2_model"

    def __post_init__(self):
        os.makedirs(self.output_dir_student1, exist_ok=True)
        os.makedirs(self.output_dir_student2, exist_ok=True)

class ReasoningExample:
    def __init__(self, data: Dict):
        self.question = data["question"]
        self.reasoning = data["reasoning"]
        self.predict_answer = float(data["predict_answer"])
        self.correct_answer = float(data["correct_answer"])
        self.is_correct = abs(self.predict_answer - self.correct_answer) < 1e-6
        self.approach_info = data.get("approach_info", {})

class ConsistencyProcessor:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def filter_dataset(self, examples: List[ReasoningExample]) -> List[ReasoningExample]:
        # Group by questions
        question_groups = {}
        for ex in examples:
            if ex.question not in question_groups:
                question_groups[ex.question] = []
            question_groups[ex.question].append(ex)
        
        # Process each group
        filtered = []
        for group in question_groups.values():
            filtered.extend(self._process_group(group))
            
        return filtered
    
    def _process_group(self, group: List[ReasoningExample]) -> List[ReasoningExample]:
        # Group by predicted answers
        answer_groups = {}
        for ex in group:
            key = round(ex.predict_answer, 6)
            if key not in answer_groups:
                answer_groups[key] = []
            answer_groups[key].append(ex)
            
        # Find most consistent group
        largest_group = max(answer_groups.values(), key=len)
        ratio = len(largest_group) / len(group)
        
        if ratio >= self.threshold:
            return largest_group
        return []