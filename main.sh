MODEL_TYPE="vllm" # openai / vllm 
MODEL_NAME="llama3.2_3B" # gpt-4o-mini / llama3.1_8B / llama3.2_3B / Qwen2.5_7B 

METHOD='metaspo'
DOMAIN='amazon'

# MetaSPO Training
python meta_train.py --method $METHOD --init_system_prompt_path "./prompts/default.json" --log_dir "./logs/$METHOD/$DOMAIN" --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" 
# This will save the optimized system prompt in "./logs/$METHOD/$DOMAIN/bilevel_nodes_0.json"

# Unseen Generalization with optimized system prompt
python meta_test.py --analysis_method 'unseen_generalization' --init_system_prompt_path "./logs/$METHOD/$DOMAIN/bilevel_nodes_0.json" --log_dir ./logs/$METHOD/unssen_generalization/$DOMAIN --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" 

# Test-Time Adaptation with optimized system prompt
python meta_test.py --analysis_method 'test_time_adaptation' --init_system_prompt_path "./logs/$METHOD/$DOMAIN/bilevel_nodes_0.json" --log_dir ./logs/$METHOD/test_time_adaptation/$DOMAIN --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" 
