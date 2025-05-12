# System Prompt Optimization with Meta-Learning
[![Paper](https://img.shields.io/badge/arXiv-2505)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://www.python.org/downloads/release/python-310s0/)
[![GCC](https://img.shields.io/badge/gcc-9.1%2B-blue)](https://gcc.gnu.org/gcc-9/)

![MetaSPO](asset/main_fig.jpg)

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance.

## Overview
This repository provides an implementation of **Meta-level System Prompt Optimizer (MetaSPO)**, a meta-learning approach for optimizing system prompts for Large Language Models (LLMs). MetaSPO focuses on improving system prompts 
that are robust to diverse user prompts and transferable across different tasks and domains.


## Running MetaSPO
### Installation
```bash
cd MetaSPO
conda create -n metaspo python=3.10 -y
conda activate metaspo
pip install -r requirements.txt
```
Ensure your OPENAI_API_KEY is stored in the .env file.

### MetaSPO: Training and Evaluation
```bash
./main.sh
```
Refer to `main.sh` for detailed instructions.


### Tasks
Modify `configs/$DOMAIN.yaml` to set dataset configurations.  
To implement new tasks, include the task name in `srt/tasks/__init__.py` and implement a corresponding task class.