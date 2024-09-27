mkdir -p tmp ; cd tmp
BASE_PATH=..

model_name=meta-llama/Llama-2-7b-chat-hf
model_name=/mnt/hwfile/llm-safety/models/Llama-2-7b-chat-hf
scorer_name=tulu-2-dpo-7b
base_output_path=${BASE_PATH}/output/alpaca_leaderboard/${model_name}/${scorer_name}-guidance
seed=1


# Here this is example of running the model with explicit value function(tokenrm) and no token wise sampling
# Need about 10hr, so you can run it on few examples
scorer_type=explicit
scorer_beta=1.0
token_wise_sampling_type=none
token_wise_sampling_beta=1

world_size=1
rank=0

output_path=${base_output_path}/bt_eft/tokenrm_b2c2l30_none1_seed1/gen
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/scripts/alpaca_leaderboard/gen_bt_eft.py \
    --model_name=${model_name} \
    --scorer_name=${scorer_name} \
    --output_path=${output_path} \
    --generation_configs.scorer_type=${scorer_type} \
    --generation_configs.temperature=0.7 \
    --generation_configs.top_p=1.0 \
    --generation_configs.beta=${scorer_beta} \
    --generation_configs.num_beams=2 \
    --generation_configs.num_candidates=2 \
    --generation_configs.block_len=30 \
    --tokenwisesampling_generation_configs.token_wise_sampling_type=${token_wise_sampling_type} \
    --tokenwisesampling_generation_configs.beta=${token_wise_sampling_beta} \
    --rank=${rank} \
    --world_size=${world_size}
