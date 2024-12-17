# Math Reasoning with Chain-of-Thought Distillation

Framework for teaching mathematical reasoning to small language models through knowledge distillation from larger models using chain-of-thought prompting.

## Overview
- Uses large language models as "teachers" to generate reasoning examples
- Distills knowledge into smaller student models through fine-tuning
- Two-stage knowledge distillation with self-consistency mechanisms
- State-of-the-art performance for small models (<2B parameters)

## Repository Structure

- `datasets/`: Contains training and evaluation datasets
- `models/`: Model architecture and weights
- `config.py`: Configuration settings
- `data_prepare.py`: Data preprocessing utilities
- `data_comb.py`: Data combination utilities
- `data_student1.py`: Data generation for labelling training
- `generate_reasoning_label.py`: Script for generating reasoning labels
- `inference.py`: Model inference pipeline
- `student1_trainer.py`: Training script for student model 1 
- `student2_trainer.py`: Training script for student model 2
- `eval_baseline.py`: Baseline model evaluation script
- `eval_student1.py`: Stage 1 evaluation metrics
- `tuned_student1_demo.ipynb`: Demo notebook for tuned student model
- `demo.ipynb`: Demo notebook for dual students inference

## Setup
### Install Requirements
```bash
pip install -r requirements.txt
```
### Prepare Dataset and Models
```bash
# (Follow the instructions in config.py to set up data and model paths)
```
### Run Training for Student Model 1
```bash
python student1_trainer.py
```
### Run Training for Student Model 2
```bash
python student2_trainer.py
```
Remark: The data with score labeled is not attached due to the cloud storage constraints.


## Usage
See tuned_student1_demo.ipynb for example usage and inference with trained student1 models. And demo.ipynb to see dual students inference.
![image](https://github.com/user-attachments/assets/6015e731-d636-40c7-af17-d9cf8f9695fc)

