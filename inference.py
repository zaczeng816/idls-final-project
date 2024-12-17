import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm

def load_models():
    # Load generator model (base model with LoRA for generation)
    print("Loading generator model...")
    BASE_MODEL_DIR = "google/gemma-2b"
    LORA_ADAPTER_DIR = "models/gemma-math-reasoning-lora-final"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        device_map="auto"
    )
    generator = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    generator = generator.merge_and_unload()
    generator.eval()
    
    # Load scorer model
    print("Loading scorer model...")
    scorer = AutoModelForCausalLM.from_pretrained(
        "models/student2_model/checkpoint-2000",
        trust_remote_code=True,
        device_map="auto"
    )
    scorer.eval()
    
    return generator, scorer, tokenizer

def generate_answers(model, tokenizer, question, num_samples=5, max_length=512):
    answers = []
    
    prompt = f"""Question: {question}
Let's solve this step by step:"""
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    for _ in range(num_samples):
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the answer
        answer = answer.replace(prompt, "").strip()
        answers.append(answer)
        
    return answers

def score_answer(model, tokenizer, question, answer):
    prompt = f"""Question: {question}
Reasoning: {answer}
Rate how likely this reasoning is correct:"""
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # Get score from the final token probabilities
    final_logits = logits[0, -1]
    score = torch.softmax(final_logits, dim=0).max().item()
    
    return score

def main():
    # Load models
    generator, scorer, tokenizer = load_models()
    
    # Load test questions
    print("Loading test data...")
    with open("datasets/test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
        
    results = []
    print("Processing questions...")
    for item in tqdm(test_data):
        question = item["question"]
        true_answer = item["answer"]
        
        # Generate multiple answers
        try:
            candidate_answers = generate_answers(generator, tokenizer, question)
            
            # Score each answer
            scores = []
            for answer in candidate_answers:
                score = score_answer(scorer, tokenizer, question, answer)
                scores.append(score)
                
            # Select best answer
            best_idx = np.argmax(scores)
            best_answer = candidate_answers[best_idx]
            
            result = {
                "question": question,
                "answer": true_answer,
                "predict_answer": best_answer
            }
            
        except Exception as e:
            print(f"Error processing question: {e}")
            result = {
                "question": question,
                "answer": true_answer,
                "predict_answer": "Error generating answer"
            }
            
        results.append(result)
        
        # Save results periodically (every 50 questions)
        if len(results) % 50 == 0:
            with open("test_results.json", "w") as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    print("Saving final results...")
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Done! Results saved to test_results.json")

if __name__ == "__main__":
    main()