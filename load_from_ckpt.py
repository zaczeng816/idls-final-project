from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Path to your checkpoint
checkpoint_path = "./student2_model/checkpoint-1500"  # Using the latest checkpoint

# First, load the PEFT config to get the base model name
peft_config = PeftConfig.from_pretrained(checkpoint_path)
base_model_name = peft_config.base_model_name_or_path

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir='../hf_models',
)

# Load the LoRA weights on top of the base model
model = PeftModel.from_pretrained(
    base_model,
    checkpoint_path,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)