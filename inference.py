import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from typing import List, Dict, Tuple
import numpy as np

class ReasoningPipeline:
    def __init__(self, config: Config):
        # Load student1 (reasoning generator)
        self.student1_tokenizer = AutoTokenizer.from_pretrained(config.output_dir_student1)
        self.student1_model = AutoModelForCausalLM.from_pretrained(config.output_dir_student1)
        
        # Load student2 (answer predictor)
        self.student2_tokenizer = AutoTokenizer.from_pretrained(config.output_dir_student2)
        self.student2_model = AutoModelForCausalLM.from_pretrained(config.output_dir_student2)
        
        self.config = config
        
    def generate_reasoning(self, question: str, num_paths: int = 5) -> List[str]:
        """Generate multiple reasoning paths"""
        prompt = f"Question: {question}\nGenerate step-by-step reasoning:"
        
        reasonings = []
        for _ in range(num_paths):
            inputs = self.student1_tokenizer(prompt, return_tensors="pt")
            
            # Generate with temperature sampling
            outputs = self.student1_model.generate(
                **inputs,
                max_length=self.config.max_seq_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            reasoning = self.student1_tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasonings.append(reasoning.split("Generate step-by-step reasoning:")[-1].strip())
            
        return reasonings
    
    def predict_answer(self, question: str, reasoning: str) -> Tuple[float, float]:
        """Predict answer with confidence score"""
        prompt = f"Question: {question}\nReasoning:\n{reasoning}\nTherefore, the answer is:"
        
        inputs = self.student2_tokenizer(prompt, return_tensors="pt")
        
        # Generate answer
        outputs = self.student2_model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        answer_text = self.student2_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        answer_text = answer_text.split("Therefore, the answer is:")[-1].strip()
        
        try:
            answer = float(answer_text)
        except:
            answer = 0.0
            
        # Calculate confidence from output scores
        confidence = float(torch.mean(torch.stack(outputs.scores)).item())
        
        return answer, confidence
    
    def solve(self, question: str) -> Dict:
        """Complete solving pipeline with self-consistency"""
        # Generate multiple reasoning paths
        reasonings = self.generate_reasoning(question, num_paths=5)
        
        # Get predictions and confidences
        predictions = []
        for reasoning in reasonings:
            answer, confidence = self.predict_answer(question, reasoning)
            predictions.append({
                "reasoning": reasoning,
                "answer": answer,
                "confidence": confidence
            })
            
        # Apply self-consistency
        answers = [p["answer"] for p in predictions]
        unique_answers = np.unique(answers)
        
        # Count occurrences and get confidences for each unique answer
        answer_stats = {}
        for ans in unique_answers:
            occurrences = sum(1 for x in answers if abs(x - ans) < 1e-6)
            confidences = [p["confidence"] for p in predictions 
                         if abs(p["answer"] - ans) < 1e-6]
            
            answer_stats[ans] = {
                "count": occurrences,
                "avg_confidence": np.mean(confidences)
            }
            
        # Select answer with highest count and confidence
        final_answer = max(
            answer_stats.items(),
            key=lambda x: (x[1]["count"], x[1]["avg_confidence"])
        )[0]
        
        # Get best reasoning for final answer
        best_prediction = max(
            [p for p in predictions if abs(p["answer"] - final_answer) < 1e-6],
            key=lambda x: x["confidence"]
        )
        
        return {
            "question": question,
            "reasoning": best_prediction["reasoning"],
            "answer": final_answer,
            "confidence": best_prediction["confidence"]
        }

def main():
    config = Config()
    pipeline = ReasoningPipeline(config)
    
    # Example usage
    question = "Bob can shuck 10 oysters in 5 minutes. How many oysters can he shuck in 2 hours?"
    result = pipeline.solve(question)
    
    print(f"Question: {result['question']}")
    print(f"\nReasoning:\n{result['reasoning']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()