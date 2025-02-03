# Bilevel System Prompt Optimization with Meta-Learning
![Method](asset/main_fig.jpg)

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. Yet, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaption even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance.

## Overview
This repository provides an implementation of **Meta-level System Prompt Optimization (MetaSPO)**, a meta-learning approach for optimizing system prompts for Large Language Models (LLMs). The method focuses on improving system prompts that are robust to diverse user prompts and transferable across different tasks and domains.

## Installation

Install dependencies using:
```bash
conda create -n metaspo python=3.9 
conda activate metaspo
pip install -r requirements.txt
```

## Usage

### Running MetaSPO Training
To train the system prompt using MetaSPO, run the following command:
```bash
python meta_train.py --config "configs/$DOMAIN.yaml" --init_system_prompt_path "./prompts/default.json" --log_dir $LOG_DIR --method 'metaspo'
```
This will save the optimized system prompt in $LOG_DIR/bilevel_nodes_0.json (last node)

### Unseen Generalization
After training, evaluate the optimized system prompt on 10 unoptimized user prompts:
```bash
python meta_test.py \
  --config "configs/$DOMAIN.yaml" \
  --init_system_prompt_path "logs/$MODEL_NAME/metaspo/$DOMAIN/bilevel_nodes_0.json" \
  --log_dir "logs/$MODEL_NAME/metaspo/$DOMAIN" \
  --method 'unseen_generalization'
```

### Test-Time Adaptation
For test-time adaptation experiment, run:
```bash
python meta_test.py \
  --config "configs/$DOMAIN.yaml" \
  --init_system_prompt_path "logs/$MODEL_NAME/metaspo/$DOMAIN/bilevel_nodes_0.json" \
  --log_dir "logs/$MODEL_NAME/metaspo/$DOMAIN" \
  --method 'apo' \
  --iteration 6
```

## Configuration
Modify `configs/$DOMAIN.yaml` to set dataset configurations
Modify `configs/base_config.yaml` to set hyperparameters, and training settings.