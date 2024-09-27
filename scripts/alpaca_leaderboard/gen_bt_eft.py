import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from inference_time_alignment.utils import set_seeds, extract_responses
from inference_time_alignment.decoder.decoder import EFTPosthocGenerationMixin, TokenWiseSamplingMixin, BeamTuningWithEFTPosthocGenerationMixin
from inference_time_alignment.scorer import ImplicitRewardScorer, ExplicitRewardScorer, NotImplementedScorer
from inference_time_alignment.model import PrefixPreTrainedWrapper
from inference_time_alignment.modeling_llama import LlamaModelForTokenScore
from inference_time_alignment.modeling_gpt2 import GPT2ModelForTokenScore
from scripts.alpaca_leaderboard.src.utils import get_chat_prompt_template, get_scorer


from utils import load_instruction_dataset, get_local_model_name

token_reward_model_template = "\n\nHuman: {raw_prompt}\n\nAssitant:"
eft_template = None

class ModelManager:
    def __init__(self):
        self.models = {}
        self.device_map = "auto"
        self.torch_dtype = torch.bfloat16

    def load_model(self, model_name, model_type, **kwargs):
        if model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            kwargs.update({"sliding_window": 4096})

        if model_name not in self.models:
            model = model_type.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                **kwargs,
            )
            self.models[model_name] = model
        return self.models[model_name]


@dataclass
class BeamTuningGenConfig:
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    # new
    num_beams: Optional[int] = 2
    num_candidates: Optional[int] = 4
    block_len: Optional[int] = 30
    # scorer
    beta: Optional[float] = 1
    average_log_prob: Optional[bool] = False
    reference_free: Optional[bool] = False
    sft_free: Optional[bool] = False
    scorer_type: Optional[str] = "implicit" # options: implicit, explicit， none
    def __post_init__(self):
        valid_types = {"implicit", "explicit", "none"}
        if self.scorer_type not in valid_types:
            raise ValueError(f"Invalid scorer type: {self.scorer_type}. Valid options are: {valid_types}")

@dataclass
class TokenWiseSamplingGenConfig:
    beta: Optional[float] = 1.0
    token_wise_sampling_type: Optional[str] = "eft"
    def __post_init__(self):
        valid_types = {"eft", "tokenrm", "none"}
        if self.token_wise_sampling_type not in valid_types:
            raise ValueError(f"Invalid token_wise sampling type: {self.token_wise_sampling_type}. Valid options are: {valid_types}")



@dataclass(kw_only=True)
class ScriptArguments:
    model_name: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    scorer_name: str = field(default="zephyr-7b-beta")
    explicit_model_type: Optional[str] = field(default=None) # options: None, ultrafeedback
    dataset_name: str = field(default="tatsu-lab/alpaca_eval")
    output_path: Text = field(default="tmp/summarize_from_feedback/gen_bt_eft.jsonl")
    overwrite: Optional[bool] = field(default=False)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=1)
    sanity_check: Optional[bool] = field(default=False)
    load_in_4bit: Optional[bool] = field(default=False)
    generation_configs: BeamTuningGenConfig = field(default_factory=lambda: BeamTuningGenConfig())
    tokenwisesampling_generation_configs: TokenWiseSamplingGenConfig = field(default_factory=lambda: TokenWiseSamplingGenConfig())

script_args = tyro.cli(ScriptArguments)
print("script_args:", script_args)
set_seeds(script_args.seed)
assert not (script_args.generation_configs.reference_free and script_args.generation_configs.sft_free)

# init datasets
dataset = load_instruction_dataset(script_args.dataset_name, script_args)
if os.path.exists(script_args.output_path) and not script_args.overwrite:
    exit()



model_manager = ModelManager()

base = model_manager.load_model(
    get_local_model_name(script_args.model_name),
    AutoModelForCausalLM,
    use_flash_attention_2=True,
    load_in_4bit=script_args.load_in_4bit
)

tokenizer = AutoTokenizer.from_pretrained(get_local_model_name(script_args.model_name))
prompt_template = get_chat_prompt_template(script_args.model_name, tokenizer)

base.generation_config.eos_token_id = None
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def get_tokenrm_scorer(beta, ):
    model_path = get_local_model_name("llama-instruction-following-token-rm")
    token_reward_model_template = "\n\nHuman: {raw_prompt}\n\nAssitant:"
    print("tokenrm model path:", model_path)
    model = model_manager.load_model(
        model_path,
        LlamaModelForTokenScore,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    explicit_reward_scorer = ExplicitRewardScorer(
        token_reward_model=model,
        tokenizer=tokenizer,
        model_prompt_template= token_reward_model_template,
        beta=beta,
    )
    return explicit_reward_scorer

if script_args.generation_configs.scorer_type == "implicit":
    scorer = get_scorer(
        scorer_name=script_args.scorer_name,
        beta=script_args.generation_configs.beta,
        average_log_prob=script_args.generation_configs.average_log_prob,
        reference_free=script_args.generation_configs.reference_free,
        load_in_4bit=script_args.load_in_4bit,
    )
    eft_template = scorer.model_prompt_template
elif script_args.generation_configs.scorer_type == "explicit":
    scorer = get_tokenrm_scorer(
        beta=script_args.generation_configs.beta,
    )
else :
    scorer = NotImplementedScorer()

if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
    if script_args.generation_configs.scorer_type != "implicit":
        eft_scorer = get_scorer(
            scorer_name=script_args.scorer_name,
            load_in_4bit=script_args.load_in_4bit,
            beta=script_args.generation_configs.beta,
            average_log_prob=script_args.generation_configs.average_log_prob,
            reference_free=script_args.generation_configs.reference_free,
        )
    else :
        eft_scorer = scorer
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
    model_path = get_local_model_name("llama-instruction-following-token-rm")
    token_reward_model_template = "\n\nHuman: {raw_prompt}\n\nAssitant:"
    token_reward_model = model_manager.load_model(
        model_path,
        LlamaModelForTokenScore,
    )
    token_reward_model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_reward_model_tokenizer.pad_token = token_reward_model_tokenizer.eos_token
    token_reward_model_tokenizer.padding_side = "left"
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "none":
    pass
else :
    raise ValueError(f"Invalid token_wise_sampling_type: {script_args.tokenwisesampling_generation_configs.token_wise_sampling_type}")


# sample
results = []
for raw_prompt, ds_id in tqdm.tqdm(zip(dataset["raw_prompt"], dataset['dataset'])):
    if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
        model = EFTPosthocGenerationMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, prompt_template.format(raw_prompt=raw_prompt)),
            tune_r=PrefixPreTrainedWrapper(eft_scorer.model,     eft_scorer.tokenizer, eft_scorer.model_prompt_template.format(raw_prompt=raw_prompt)),
            base_r=PrefixPreTrainedWrapper(eft_scorer.ref_model, eft_scorer.tokenizer, eft_scorer.model_prompt_template.format(raw_prompt=raw_prompt)),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
        max_prefix_input_ids_length= max(model.base.get_prefix_input_ids_length(), model.tune_r[0].get_prefix_input_ids_length(), model.base_r.get_prefix_input_ids_length())
        max_new_tokens = script_args.generation_configs.max_length - max_prefix_input_ids_length
    elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
        model = TokenWiseSamplingMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, prompt_template.format(raw_prompt=raw_prompt)),
            token_reward_model=PrefixPreTrainedWrapper(token_reward_model, token_reward_model_tokenizer, token_reward_model_template.format(raw_prompt=raw_prompt)),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
        max_prefix_input_ids_length= max(model.base.get_prefix_input_ids_length(), model.token_reward_model.get_prefix_input_ids_length())
        max_new_tokens = script_args.generation_configs.max_length - max_prefix_input_ids_length
    else :
        model = PrefixPreTrainedWrapper(base, tokenizer, prompt_template.format(raw_prompt=raw_prompt))
        max_new_tokens = script_args.generation_configs.max_length - model.get_prefix_input_ids_length()
    
    bt_model = BeamTuningWithEFTPosthocGenerationMixin(model, tokenizer)
    
    prompt = ""
    prompt_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    outputs = bt_model.bon_beam_sample(
        input_ids=prompt_tokenized["input_ids"].cuda(),
        attention_mask=prompt_tokenized["attention_mask"].cuda(),
        scorer=scorer.set_raw_prompt(raw_prompt),
        split_by_prompt_text=False,
        # 
        max_length=max_new_tokens,
        temperature=script_args.generation_configs.temperature,
        top_p=script_args.generation_configs.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        # 
        num_beams=script_args.generation_configs.num_beams,
        num_candidates=script_args.generation_configs.num_candidates,
        block_len=script_args.generation_configs.block_len, # block_len = inf to enable sequence wise bon
        # return_dict_in_generate=True,
    )
    response = extract_responses(outputs, tokenizer, prompt_len=prompt_tokenized["input_ids"].size(1))[0]
    results.append({
        "instruction": raw_prompt,
        "output": response,
        "generator": f"{script_args.model_name}({str(script_args)})",
        "datasplit": ds_id,
        "split": "eval"
    })
    print("prompt:", raw_prompt)
    print("response:", response)

dataset = Dataset.from_list(results)
dataset.to_json(script_args.output_path)
