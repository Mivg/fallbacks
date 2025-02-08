import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loop_generation.structs import Loop
from typing import List
from utils.logger import get_logger


# get logger
logger = get_logger()


def read_jsonl_to_list(file_path: str):
    """
    read a jsonl path to a list of jsons
    used for example to parse qampari data
    """
    # read path
    with open(file_path, 'r') as json_file:
        return list(map(json.loads, json_file))


def load_model_and_tokenizer(model_name: str):
    """
    get a huggingface model and tokenize from a model name
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def dump_loops_to_jsonl(loops: List[Loop],
                         output_path: str) -> None:
    """
    dumps a list of loop objects to a jsonl file
    """
    # log
    num_loops = len(loops)
    logger.info(f"Writing {num_loops} examples to {output_path}.")
    with open(output_path, 'w') as f:
        for l in loops:
            json.dump(
                {
                    "text": l.text,
                    "prompt": l.prompt,
                    "confidence": l.confidence,
                },
                f)
            f.write('\n')
