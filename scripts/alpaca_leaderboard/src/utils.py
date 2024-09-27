import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_time_alignment.scorer import ImplicitRewardScorer
from utils import get_local_model_name

def get_scorer(scorer_name, load_in_4bit, beta, average_log_prob, reference_free):
    if scorer_name == "ultrafeedback-llama-2-7b":
        model = AutoModelForCausalLM.from_pretrained(
            get_local_model_name("llama-instruction-following-dpo"),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            get_local_model_name("llama-instruction-following"),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("llama-instruction-following"))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        prompt_template = "<s>[INST] {raw_prompt} [/INST] "
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            model_prompt_template=prompt_template,
            ref_model_prompt_template=prompt_template,
            beta=beta,
            average_log_prob=average_log_prob,
            reference_free=reference_free,
        )
        return implicit_reward_scorer
    elif scorer_name == "zephyr-7b-beta":
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/mistral-7b-sft-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        prompt_template = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            model_prompt_template=prompt_template,
            ref_model_prompt_template=prompt_template,
            beta=beta,
            average_log_prob=average_log_prob,
            reference_free=reference_free,
        )
        return implicit_reward_scorer
    elif scorer_name == "Starling-LM-7B-alpha":
        model = AutoModelForCausalLM.from_pretrained(
            "berkeley-nest/Starling-LM-7B-alpha",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            "openchat/openchat_3.5",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        prompt_template = tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        ) + " "
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            model_prompt_template=prompt_template,
            ref_model_prompt_template=prompt_template,
            beta=beta,
            average_log_prob=average_log_prob,
            reference_free=reference_free,
        )
        return implicit_reward_scorer
    elif scorer_name == "tulu-2-dpo-7b":
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/tulu-2-dpo-7b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            "allenai/tulu-2-7b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )
        tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        prompt_template = tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            model_prompt_template=prompt_template,
            ref_model_prompt_template=prompt_template,
            beta=beta,
            average_log_prob=average_log_prob,
            reference_free=reference_free,
        )
        return implicit_reward_scorer
    else:
        raise NotImplementedError


def get_evaluator(evaluator_name):
    from scripts.alpaca_leaderboard.src.evaluator.starlingrm import StarlingRMEvaluator
    from scripts.alpaca_leaderboard.src.evaluator.ultrarm import UltraRMEvaluator
    if evaluator_name == "Starling-RM-34B":
        return StarlingRMEvaluator()
    elif evaluator_name == "UltraRM-13b":
        return UltraRMEvaluator()
    else:
        NotImplementedError


def get_chat_prompt_template(model_name, tokenizer):
    if model_name in ("meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "/mnt/hwfile/llm-safety/models/Llama-2-7b-chat-hf", "/mnt/hwfile/llm-safety/models/Llama-2-70b-chat-hf"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        ) + " " # add a trailing space
    elif model_name in ("mistralai/Mistral-7B-Instruct-v0.2","mistralai/Mixtral-8x7B-Instruct-v0.1",):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/mistral-7b-sft-beta"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True,
        )
    elif model_name in ("berkeley-nest/Starling-LM-7B-alpha", "openchat/openchat_3.5"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        ) + " " # add a trailing space
    elif model_name in ("allenai/tulu-2-dpo-7b", "allenai/tulu-2-7b"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user",   "content": "{raw_prompt}"},
            ],
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        raise NotImplementedError
