MODEL_TYPE="vllm" # openai / vllm 
MODEL_NAME="llama3.2_3B" # gpt-4o-mini / llama3.1_8B / llama3.2_3B / Qwen2.5_7B 

METHOD='metaspo'
DOMAIN='amazon'
LOG_DIR="./logs/$MODEL_NAME/$METHOD/$DOMAIN"

# MetaSPO Training
python meta_train.py --config "configs/$DOMAIN.yaml" --init_system_prompt_path "./prompts/default.json" --log_dir $LOG_DIR --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" --method $METHOD
# This will save the optimized system prompt in $LOG_DIR/bilevel_nodes_0.json (last node)

# # Unseen Generalization with optimized system prompt
# UNSEEN_GENERALIZATION='unseen_generalizatoin'
# python meta_test.py --config "configs/$DOMAIN.yaml" --init_system_prompt_path "$LOG_DIR/bilevel_nodes_0.json" --log_dir $LOG_DIR --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" --method $UNSEEN_GENERALIZATION

# # Test-Time Adaptation with optimized system prompt
# TEST_TIME_METHOD='apo'
# python meta_test.py --config "configs/$DOMAIN.yaml" --init_system_prompt_path "$LOG_DIR/bilevel_nodes_0.json" --log_dir $LOG_DIR --base_model_type "$MODEL_TYPE" --base_model_name "$MODEL_NAME" --method $TEST_TIME_METHOD --iteration 6
