import os
from dataclasses import dataclass, field
from typing import Optional, Text

import tyro
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from datasets import Dataset

from inference_time_alignment.utils import set_seeds, extract_responses, StopOnStringCriteria
from inference_time_alignment.decoder.decoder import EFTPosthocGenerationMixin, TokenWiseSamplingMixin, BeamTuningWithEFTPosthocGenerationMixin
from inference_time_alignment.scorer import ImplicitRewardScorer, ExplicitRewardScorer, NotImplementedScorer
from inference_time_alignment.model import PrefixPreTrainedWrapper
from inference_time_alignment.modeling_llama import LlamaModelForTokenScore
from inference_time_alignment.modeling_gpt2 import GPT2ModelForTokenScore, GPT2ForScore

from utils import load_instruction_dataset, get_local_model_name

DEFAULT_PROMPT_TEMPLATE = """\
SUBREDDIT: r/relationships\nTITLE: Me [21F] with my boyfriend [19M] of 2months has a gaming habit.\nPOST: I tried to start a convo with the boyfriend everyday but it seems to be making me a little depress because he's always playing video games than paying attention to me. I'm not trying to be an attention but it's seems to be a bad habit of his. I don't know what to do or how to even confront him about it. Any IDEAS?
TL;DR: Boyfriend has a gaming habit and I don't know how to confront him about it. I'm not trying to be an attention whore but it's making me feel bad.

SUBREDDIT: r/relationships\nTITLE: Me [21 M] with my...idk [19 F] Just not sure what to do.\nPOST: Went on vacation 1 1/2 years ago\n\nMet an amazing girl\n\nSpent a lot of time together\n\nHad to leave\n\nWe had agreed it would be ok to see other people\n\nBut we keep in contact and talk about how much we miss each other all the time\n\nStill have feelings for her\n\nShe just entered a relationship recently\n\nIt bothers me\n\nIdk if I should tell her how I feel or if I am just idealizing something we had and should move on.
TL;DR: Had an amazing time with this girl before we had to leave for summer vacation 1 1/2 years ago. Still have feelings for her and want to pursue relationship w/ her. Don't know whether to tell her or not.

{raw_prompt}
TL;DR:\
"""

class ModelManager:
    def __init__(self):
        self.models = {}
        self.device_map = "auto"
        self.torch_dtype = torch.bfloat16

    def load_model(self, model_name, model_type, **kwargs):
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
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    eos_strings: Optional[list] = field(default_factory=lambda: ["\n", "<|endoftext|>"])
    # new
    num_beams: Optional[int] = 4
    num_candidates: Optional[int] = 4
    block_len: Optional[int] = 5
    # scorer
    beta: Optional[float] = 1
    average_log_prob: Optional[bool] = False
    reference_free: Optional[bool] = False
    sft_free: Optional[bool] = False
    scorer_type: Optional[str] = "implicit" # options: implicit, explicit， none
    def __post_init__(self):
        valid_types = {"implicit", "explicit" , "none"}
        if self.scorer_type not in valid_types:
            raise ValueError(f"Invalid scorer type: {self.scorer_type}. Valid options are: {valid_types}")

@dataclass
class TokenWiseSamplingGenConfig:
    beta: Optional[float] = 1.0
    token_wise_sampling_type: Optional[str] = "eft"
    def __post_init__(self):
        valid_types = {"eft", "tokenrm", "none", "int", } # add int
        if self.token_wise_sampling_type not in valid_types:
            raise ValueError(f"Invalid token_wise sampling type: {self.token_wise_sampling_type}. Valid options are: {valid_types}")



@dataclass(kw_only=True)
class ScriptArguments:
    model_name: str = field(default="openai-community/gpt2-large")
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)
    dataset_name: str = field(default="chadlzx/openai_summarize_comparisons_relabel")
    output_path: Text = field(default="tmp/summarize_from_feedback/gen_bt.jsonl")
    overwrite: Optional[bool] = field(default=False)
    rank: Optional[int] = field(default=0)
    world_size: Optional[int] = field(default=1)
    seed: Optional[int] = field(default=1)
    sanity_check: Optional[bool] = field(default=False)
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

def get_eft_scorer(beta, average_log_prob, reference_free, sft_free):
    model = model_manager.load_model(
        get_local_model_name("gpt2-summarize-dpo"),
        AutoModelForCausalLM,
    )
    ref_model = model_manager.load_model(
        get_local_model_name("gpt2-summarize") if not sft_free else "openai-community/gpt2",
        AutoModelForCausalLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("gpt2-summarize"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    implicit_reward_scorer = ImplicitRewardScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        model_prompt_template="{raw_prompt}TL;DR: ",
        ref_model_prompt_template="{raw_prompt}TL;DR: ",
        beta=beta,
        average_log_prob=average_log_prob,
        reference_free=reference_free,
    )
    return implicit_reward_scorer

def get_tokenrm_scorer(beta, ):
    model_path = get_local_model_name("gpt2-summarize-token-rm")
    model = model_manager.load_model(
        model_path,
        GPT2ModelForTokenScore,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    explicit_reward_scorer = ExplicitRewardScorer(
        token_reward_model=model,
        tokenizer=tokenizer,
        model_prompt_template="{raw_prompt}TL;DR: ",
        beta=beta,
    )
    return explicit_reward_scorer

if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
    tune_gpt2_small = model_manager.load_model(
        get_local_model_name("gpt2-summarize-dpo"),
        AutoModelForCausalLM,
    )
    base_gpt2_small = model_manager.load_model(
        get_local_model_name("gpt2-summarize"),
        AutoModelForCausalLM,
    )
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
    token_reward_model = model_manager.load_model(
        get_local_model_name("gpt2-summarize-token-rm"),
        GPT2ModelForTokenScore,
    )
elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "none":
    pass
else :
    raise ValueError(f"Invalid token_wise_sampling_type: {script_args.tokenwisesampling_generation_configs.token_wise_sampling_type}")

base = model_manager.load_model(
    script_args.model_name,
    AutoModelForCausalLM,
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
base.generation_config.eos_token_id = None
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if script_args.generation_configs.scorer_type == "implicit":
    scorer = get_eft_scorer(
        beta=script_args.generation_configs.beta, 
        average_log_prob=script_args.generation_configs.average_log_prob,
        reference_free=script_args.generation_configs.reference_free,
        sft_free=script_args.generation_configs.sft_free
    )
elif script_args.generation_configs.scorer_type == "explicit":
    scorer = get_tokenrm_scorer(
        beta=script_args.generation_configs.beta,
    )
else :
    scorer = NotImplementedScorer()

# sample
results = []
for raw_prompt in tqdm.tqdm(dataset["raw_prompt"]):

    if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
        model = EFTPosthocGenerationMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt)),
            tune_r=PrefixPreTrainedWrapper(tune_gpt2_small, tokenizer, f'{raw_prompt}TL;DR: '),
            base_r=PrefixPreTrainedWrapper(base_gpt2_small, tokenizer, f'{raw_prompt}TL;DR: '),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
    elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
        model = TokenWiseSamplingMixin(
            base=PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt)),
            token_reward_model=PrefixPreTrainedWrapper(token_reward_model, tokenizer, f'{raw_prompt}TL;DR: '),
            w=script_args.tokenwisesampling_generation_configs.beta,
        )
    else :
        model = PrefixPreTrainedWrapper(base, tokenizer, script_args.prompt_template.format(raw_prompt=raw_prompt))
    
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
        top_k=None,
        # 
        max_new_tokens=script_args.generation_configs.max_new_tokens,
        temperature=script_args.generation_configs.temperature,
        eos_strings=script_args.generation_configs.eos_strings,
        # 
        num_beams=script_args.generation_configs.num_beams,
        num_candidates=script_args.generation_configs.num_candidates,
        block_len=script_args.generation_configs.block_len, # block_len = inf to enable sequence wise bon
        # return_dict_in_generate=True,
    )
    response = extract_responses(outputs, tokenizer, prompt=prompt)[0]
    for eos_string in script_args.generation_configs.eos_strings:
        response = response.split(eos_string)[0]
    results.append({
        "prompt": raw_prompt,
        "response": response,
    })
    print("prompt:", raw_prompt)
    print("response:", response)

dataset = Dataset.from_list(results)
dataset.to_json(script_args.output_path)
