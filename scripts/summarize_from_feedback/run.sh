mkdir -p tmp ; cd tmp
BASE_PATH=..

model_name=openai-community/gpt2-large
model_name=/mnt/hwfile/llm-safety/models/gpt2-large
scorer_name=
base_output_path=${BASE_PATH}/output/summarize/${model_name}
seed=1




scorer_type=explicit
scorer_beta=1.0
token_wise_sampling_type=none
token_wise_sampling_beta=1

world_size=1
rank=0

output_path=${base_output_path}/bt_eft/tokenrm_b4c4l5_none1_seed1/gen
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/scripts/summarize_from_feedback/gen_bt_eft.py \
    --model_name=${model_name} \
    --output_path=${output_path} \
    --generation_configs.num_beams=4 \
    --generation_configs.num_candidates=4 \
    --generation_configs.block_len=5 \
    --generation_configs.scorer_type=${scorer_type} \
    --generation_configs.beta=${scorer_beta} \
    --tokenwisesampling_generation_configs.token_wise_sampling_type=${token_wise_sampling_type} \
    --tokenwisesampling_generation_configs.beta=${token_wise_sampling_beta} \
    --seed=${seed} \
    --rank=${rank} \
    --world_size=${world_size}

