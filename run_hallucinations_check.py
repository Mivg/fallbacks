import argparse
import gc
import json
import os
import re
import string
from copy import deepcopy
from enum import Enum
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.degeneration_detector import Sample, analyze_sequence

ENUM_ITEM_PATTERN = re.compile(r'^(\d+)\.\s*(.*)$')
NUMBER_PATTERN = re.compile(r'\d+')

CORRECT = 1
HALLUCINATION = 2
REPETITION = 3
BAD_FORMAT = 4
TOPIC_CHANGE = 5
EOS = 6
PAD = 7
ANNOT_DICT = {CORRECT: 'C', HALLUCINATION: 'H', REPETITION: 'R'}

ICL_INSTRUCTIONS = {
    'icl': 'The following 25 are known moons of Mars\n1. Phobos\n2. Deimos\n\n'
           'The following 25 are the species of the main characters with '
           'a dialogue in the movie The Lion King\n1. Lion\n2. Warthog\n3. Meerkat\n4. Mandrill\n'
           '5. Hyena\n6. Hornbill\n\n'  # https://chat.openai.com/share/8cbc6b27-fa35-43e4-9d8e-1dafa29ef1b6
           'The following 25 are the vegetables common in traditional greek salads\n'
           '1. Tomato\n2. Cucumber\n3. Onion\n4. Pepper\n5. Kalamata olive\n\n',
    'idk': 'Complete the following list with facts you are sure of, and stop when you cannot recall additional facts.\n'
}


class AnswerListStatus(Enum):
    FULL = -1  # extracted answers at least as many as expected and the ist continued
    BAD_FORMAT = BAD_FORMAT  # extracted not enough answers because the format was not good
    TOPIC_CHANGE = TOPIC_CHANGE  # extracted not enough answers (or exactly as many as expected) and the model changed the topic
    EOS = EOS  # extracted not enough answers (or exactly as many as expected) and the model declared EOS


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    assert torch.cuda.device_count() == 1
print('\t====> The device that will be used is:', device)
print('Cuda visible devices=', os.getenv('CUDA_VISIBLE_DEVICES'))


def generate_predictions(model_name, prompts, output_dir, gen_length, batch_size=8, prefix='',
                         prompt_prefix='\n1.', temperature=0, repetitions=1):
    temp_suffix = '' if temperature == 0 else f'_temp={temperature}'.replace('.', '_')
    # Note - there is no accounting for seed when caching multiple repetitions of the prompts with random sampling, so should not do it with caching for full reproducibility. Also, if the batches are different can affect the sampling
    n_prompts = int(len(prompts) / repetitions)
    print(
        f'Generating predictions for a total of {len(prompts)} prompts which are {n_prompts} unique prompts repeated {repetitions} times each')
    # due to the fact that the repetitions are concatenated lists, we can actually use cache of more/less repetitions directly
    fname = os.path.join(output_dir,
                         f"{prefix}_results_{n_prompts}_prompts_{model_name}_{gen_length}T{temp_suffix}.".replace('/',
                                                                                                                  '+') + "jsonl")
    logprobs_fname = os.path.join(output_dir,
                                  f"{prefix}_logprobs_{n_prompts}_prompts_{model_name}_{gen_length}T{temp_suffix}.".replace(
                                      '/', '+') + "jsonl")
    os.makedirs(output_dir, exist_ok=True)
    all_prompts = prompts
    cached_results, cached_logprobs = [], []
    if os.path.exists(fname) and os.path.exists(logprobs_fname):
        print(f'Found existing results file {fname}, skipping generation.')
        with open(fname, 'r') as f:
            cached_results = [json.loads(line.strip())['completion'] for line in f.readlines()]
            if len(cached_results) >= len(prompts):
                return cached_results[:len(prompts)]  # may have cached even more repetitions than we need now

            prompts = prompts[len(cached_results):]
            print(f'Found {len(cached_results)} cached results, generating the rest ({len(prompts)})')
        with open(logprobs_fname, 'r') as f:
            cached_logprobs = [json.loads(line.strip())['logprobs'] for line in f.readlines()]

    results = []
    logprobs_results = []

    if 'chat' in model_name.lower() or 'dolly' in model_name.lower() or 'instruct' in model_name.lower() or 'sft' in model_name.lower():
        _generate_predictions_local_model_with_pipeline(gen_length, logprobs_results, model_name, prompt_prefix,
                                                        prompts,
                                                        results, temp=temperature)
    else:
        _generate_predictions_local_model(batch_size, gen_length, logprobs_results, model_name, prompt_prefix, prompts,
                                          results, temp=temperature)

    # In openAI it already happened, but no harm in redoing it..
    with open(fname, 'w') as f:
        for prompt, result in zip(all_prompts, cached_results + results):
            f.write(json.dumps({"prompt": prompt, "completion": result}) + '\n')
    with open(logprobs_fname, 'w') as f:
        for prompt, logprobs_result in zip(all_prompts, cached_logprobs + logprobs_results):
            f.write(json.dumps({"prompt": prompt + prompt_prefix, "logprobs": logprobs_result}) + '\n')
    gc.collect()
    # release gpu memory (cuda)
    torch.cuda.empty_cache()

    return results


# pipelines do not work in batches! they process the inputs one by one
def get_generate_for_pipeline(pipeline, scores, top=5):
    generate = pipeline.model.generate

    def f(*args, **kwargs):
        with torch.no_grad():
            r = generate(*args, **kwargs)
            assert len(
                r.sequences) == 1, 'We assume here that the pipelines run the prompts one by one, but this does not appear to be the case'
            log_probs = log_softmax(torch.cat(r.scores).detach().cpu(), dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, top)  # Get top-5 log probabilities and their indices
            # Decode each index to its corresponding token
            tokens = [pipeline.tokenizer.batch_decode(idx) for idx in topk_indices]
            token_logprob_dict = [dict(zip(five_token, five_log_prob.tolist())) for five_token, five_log_prob in
                                  zip(tokens, topk_log_probs)]
            scores.append(token_logprob_dict)
            return r.sequences

    return f


def _generate_predictions_local_model_with_pipeline(gen_length, logprobs_results, model_name, prompt_prefix, prompts,
                                                    results, temp=0):
    clean_model_name = model_name.strip()
    pipeline_kwargs = dict(model=clean_model_name,
                           output_scores=True, do_sample=False, return_dict_in_generate=True,
                           trust_remote_code=True, max_new_tokens=gen_length, top_p=1, top_k=None,
                           device=device)  # return_type=ReturnType.NEW_TEXT
    if 'step' in clean_model_name:
        assert 'olmo' in clean_model_name.lower() or 'pythia' in clean_model_name.lower(), 'Only OLMO and Pythia models have step versions'
        clean_model_name, revision = clean_model_name.split('step')
        revision = 'step' + revision
        pipeline_kwargs['model'] = clean_model_name
        pipeline_kwargs['revision'] = revision
    elif clean_model_name in {'meta-llama/Llama-2-70b-hf', 'meta-llama/Llama-2-70b-chat-hf',
                              'meta-llama/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B-Instruct'}:
        print('loading llama 70b in 8 bit')
        pipeline_kwargs['model_kwargs'] = {'load_in_8bit': True}
        del pipeline_kwargs[
            'device']  # when the model is loaded with accelerate, it cannot be moved to a specific device

    if temp > 0:
        pipeline_kwargs['do_sample'] = True
        pipeline_kwargs['temperature'] = temp

    pipe = pipeline("text-generation", **pipeline_kwargs)
    scores = []
    pipe.model.generate = get_generate_for_pipeline(pipe, scores)  # hack to get the logits

    # pipelines do not work in batches! they process the inputs one by one
    batch = [b + prompt_prefix for b in prompts]
    batch_results = [x[0]['generated_text'] for x in pipe(batch)]
    if all(s.startswith(p) for s, p in zip(batch_results, batch)):
        print(
            f'Looks like in model {model_name} the prompt is echoed back, so removing it...')  # this is since some pipelines don't accept return_type=ReturnType.NEW_TEXT
        batch_results = [s[len(p):].lstrip() for s, p in zip(batch_results, batch)]
    results.extend(batch_results)  # batch_results should be a list of strings, just the new text
    logprobs_results.extend(scores)  # scores should be a list of lists of dicts from token (str) to its logprob


def _generate_predictions_local_model(batch_size, gen_length, logprobs_results, model_name, prompt_prefix, prompts,
                                      results, temp=0):
    # Load the model and the tokenizer
    # NOTE - no cache in local models - either the file exists, or we do inference from scratch
    clean_model_name = model_name.strip()
    if 'step' in clean_model_name:
        assert 'olmo' in clean_model_name.lower() or 'pythia' in clean_model_name.lower(), 'Only OLMO and Pythia models have step versions'
        clean_model_name, revision = clean_model_name.split('step')
        revision = 'step' + revision
        model = AutoModelForCausalLM.from_pretrained(clean_model_name, revision=revision, trust_remote_code=True)
    elif clean_model_name in {'meta-llama/Llama-2-70b-hf', 'meta-llama/Llama-2-70b-chat-hf',
                              'meta-llama/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B-Instruct'}:
        print('loading llama 70b in 8 bit')
        # https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996
        model = AutoModelForCausalLM.from_pretrained(clean_model_name, trust_remote_code=True,
                                                     load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(clean_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(clean_model_name, trust_remote_code=True, padding_side='left')
    try:
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            free_memory = torch.cuda.memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            print('*********************** Before loading the model into memory ***********************')
            print(f"Free memory: {free_memory / 1024 ** 3:.2f} GB")
            print(f"Total memory: {total_memory / 1024 ** 3:.2f} GB")

        model = model.to(device)
    except Exception as e:
        if 'memory' in str(e):
            raise RuntimeError("Cannot even load the model into the gpu so no point in trying different batch sizes...")
    model.eval()
    # Set the padding token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    for i in trange(0, len(prompts), batch_size, desc='Batches'):
        batch = prompts[i:i + batch_size]
        batch_encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=False,
                                  return_attention_mask=True)

        # Add the '\n1.' ending for each prompt in the batch
        if len(prompt_prefix) > 0:
            endings = tokenizer([prompt_prefix] * len(batch), return_tensors='pt', padding=True, truncation=True,
                                return_attention_mask=True)
            input_ids = torch.cat([batch_encoded['input_ids'], endings['input_ids']], dim=-1)
            attention_masks = torch.cat([batch_encoded['attention_mask'], endings['attention_mask']], dim=-1)
        else:
            input_ids = batch_encoded['input_ids']
            attention_masks = batch_encoded['attention_mask']

        with torch.no_grad():
            kwargs = {'input_ids': input_ids.to(device),
                      'attention_mask': attention_masks.to(device),
                      'return_dict_in_generate': True,
                      'output_scores': True,
                      'do_sample': False
                      }
            if temp > 0:
                kwargs['do_sample'] = True
                kwargs['temperature'] = temp
            # if 'olmo' not in model_name.lower():
            #    kwargs.update({'max_length': input_ids.shape[1] + gen_length, 'num_return_sequence': 1})
            # else:
            kwargs.update({'max_new_tokens': gen_length, 'top_k': None, 'top_p': 1.})
            outputs = model.generate(**kwargs)
            # do_sample True will change from greedy decoding, but False is the default in HF
            # we can also set num_beams to something other than 1 (default) to use beam search
            # temperature is default to 1 as well
            # top_k, top_p, repetition_penalty, length_penalty all default to 1.0

        # The following includes the input ids..
        # batch_results = [tokenizer.decode(output.cpu()).strip()[len(prompt)+1:] for prompt, output in zip(batch, outputs.sequences)]

        # Extract only the generated part
        generated_sequences = outputs.sequences[:, input_ids.shape[1]:].detach().cpu()

        # If you want to decode them into text
        batch_results = [tokenizer.decode(generated_seq, skip_special_tokens=True) for generated_seq in
                         generated_sequences]

        results.extend(batch_results)

        # for each sample, the logprobs is a list of the length of the completed output logprobs,
        # each is a dict of size 5 from the token value (in string) to its logprob (float)
        # assert batch_size == 1, 'not sure the following will work otherwise'

        for i in range(len(batch)):
            top_k_token_log_probs = []
            for score in outputs.scores:  # List of tensors with logits:
                log_probs = log_softmax(score[i].detach().cpu(), dim=-1)  # Convert logits to log probabilities
                topk_log_probs, topk_indices = torch.topk(log_probs, 5)  # Get top-5 log probabilities and their indices

                # Decode each index to its corresponding token
                tokens = [tokenizer.decode([idx]) for idx in topk_indices]

                # Create a dictionary for the current sequence
                token_logprob_dict = {token: log_prob.item() for token, log_prob in zip(tokens, topk_log_probs)}
                top_k_token_log_probs.append(token_logprob_dict)

            logprobs_results.append(top_k_token_log_probs)


MINIMUM_ANSWERS = 3


def extract_answers(result, require_max_answers=False, require_enumerated_answers=False, max_answers=500,
                    max_words_for_eos=150, return_before_after=False):
    # first, let's see if the model decided to not use \n but did use proper enumeration
    topic_change = eos = trimmed = False
    n_words = len(result.split())

    # trying to find the list in the first line
    first_line = result.split('\n')[0]
    answers = None
    if len(enumerators := re.findall(r'(\d+)\.\s?\w', first_line)) >= MINIMUM_ANSWERS:
        if all(int(enumerators[i]) == i + 2 for i in range(len(enumerators))):
            print('Found a match in a non-new line template!')
            answers = re.split(r'\d+\.\s?', first_line)
            # re -adding enumerations as later checks expect it
            answers = [f'{i}. {ans.strip()}' for i, ans in enumerate(answers,
                                                                     2)]  # TODO - don't we expect the first answer to appear without the prefix \n1. ?
            topic_change = len(answers) <= max_answers
            eos = len([l for l in result.split('\n') if len(l.strip()) > 0]) == 1  # only this single line
            trimmed = True
        else:
            # print('ah?')
            pass
    elif len(parts := first_line.split(',')) >= MINIMUM_ANSWERS:
        # first have to verify there is no enumeration in later lines
        if len(enumerators := re.findall(r'(\d+)\.\s*\w', result)) >= MINIMUM_ANSWERS:
            print(
                'Despite finding a csv in first line, it looks like there is enumeration later on so not treating as csv')
        else:
            print('Found a match as a CSV in the first line!')
            answers = [f'{i}. {ans.strip()}' for i, ans in enumerate(parts, 2)]
            topic_change = len(answers) <= max_answers
            eos = len([l for l in result.split('\n') if len(l.strip()) > 0]) == 1  # only this single line
            trimmed = True

    if answers is None:  # i.e. could not find the result in the first line
        # assumption - each answer is in a separate line, starting with \d+\.

        # in some cases, this break in templates such as 1.\nSOME TEXT\n2.\nSOME OTHER TEXT or even
        # 1.\n\nSOME TEXT\n2.\n\nSOME OTHER TEXT so we want to normalize it
        pattern = r'\n(\d+)\.\s(?=\D)+'
        replacement = r'\n\1. '
        result_ = re.sub(pattern, replacement, result)
        if result_ != result:
            print('Detected a weird pattern of multiple spaces or break lines after enumerations. normalizing it...')
            result = result_

        # first, see if the model decided to create its own lists
        if '\nThe following ' in result:
            print(
                'Found a suspected divergence from template to new lists due to "The following..." pattern. Skipping everything after it')
            after = result[result.index('\nThe following '):]
            before = result = result.split('\nThe following ')[0]
            trimmed = True
            topic_change = True
            eos = False
        if '\n\n' in result:
            before = result_up_to_newlines = result[:result.index('\n\n')]
            after = result[result.index('\n\n'):]
            if result_up_to_newlines.split('\n')[-1].startswith(f'{max_answers}.'):
                print(
                    'Found a suspected divergence from template to new lists due to \\n\\n appearance. Skipping everything after it')
                topic_change = True
                eos = len(result[result.index('\n\n'):].strip()) == 0  # there was no more content afterwards
                trimmed = not eos
                result = result_up_to_newlines
            else:
                probably_divergence = True
                for i, line in enumerate(result_up_to_newlines.split('\n')[1:], 2):
                    match = ENUM_ITEM_PATTERN.match(line)
                    if match is None or int(match.group(1)) != i:
                        probably_divergence = False
                if probably_divergence and len(result_up_to_newlines.split('\n')) >= MINIMUM_ANSWERS:
                    print(
                        f'Found a suspected divergence from template to new lists due to \\n\\n appearance but with less than {max_answers} answers. Skipping everything after it')
                    topic_change = True
                    eos = len(result[result.index('\n\n'):].strip()) == 0  # there was no more content afterwards
                    trimmed = not eos
                    result = result_up_to_newlines

        # answers = [l for l in result.lstrip().split('\n') if len(l) > 0][:max_answers]
        answers = [l for l in result.lstrip().split('\n') if len(l) > 0]  # account for EOS in this scenario

        if len(answers) < max_answers:
            print(f'Found only {len(answers)} answers instead of {max_answers}')
            bad_format = '\n2.' not in result
            if bad_format:
                print(f'\t-- The result seems to be in a bad format: {result}')
            if require_max_answers:
                raise ValueError(f'Found only {len(answers)} answers instead of {max_answers}')

    answers_list = []
    for i, ans in enumerate(answers, 1):
        match = ENUM_ITEM_PATTERN.match(ans)
        if require_enumerated_answers:
            # TODO - this would fail now because the results we get here are without the prefix \n1.
            assert match is not None, f'Found an answer that does not start with a number: {ans}'
            assert int(match.group(1)) == i, f'Found an answer that does not start with the correct number: {ans}'
        if match is not None:
            answers_list.append(match.group(2))
        else:
            if re.match('^\d+\.?\s?$', ans) is not None and i == len(answers):
                # probably the last answer was just clipped too short, not a bad template
                print(f'Found a last answer that is just a number ({ans}), treating as missing (skipping)')
                eos = False
                continue

            print(f'Found an answer that does not start with a number ({ans}), adding it as is')
            answers_list.append(ans)
    topic_change = topic_change or len(answers) <= max_answers  # i.e. it stopped creating content
    eos = eos or (not trimmed and len(answers) <= max_answers)
    if eos and n_words > max_words_for_eos:
        print(
            'Detected what looked like a topic change or EOS but due to number of words phrasing it as bad format instead')
        topic_change = eos = False

    if max_answers is not None and len(answers_list) > max_answers:
        print(f'Found more than {max_answers}, so clipping results')
        answers_list = answers_list[:max_answers]

    topic_change = topic_change and len(answers_list) >= MINIMUM_ANSWERS
    if len(answers_list) < max_answers:
        if not topic_change:
            answer_status = AnswerListStatus.BAD_FORMAT
        elif eos:
            answer_status = AnswerListStatus.EOS
        else:
            answer_status = AnswerListStatus.TOPIC_CHANGE
    else:
        answer_status = AnswerListStatus.FULL if not topic_change else AnswerListStatus.TOPIC_CHANGE
    if return_before_after:
        if answer_status == AnswerListStatus.TOPIC_CHANGE:
            assert before is not None
            return answers_list, answer_status, before, after
        return answers_list, answer_status, None, None
    return answers_list, answer_status


def normalize_answer(s, should_remove_year=False):
    """Lower text and remove punctuation, articles and extra whitespace."""

    # from https://huggingface.co/datasets/tau/scrolls/blob/main/metrics/exact_match.py
    # seems to match this https://github.com/samsam3232/qampari/blob/master/models/evaluation/reader_metrics.py

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_year(text):
        # added by me
        return re.sub(r"\(\d{4}\)", "", text).strip()

    if should_remove_year:
        s = remove_year(s)

    return white_space_fix(remove_articles(remove_punc(remove_trailing_parenthesis(lower(s).strip()))))


from typing import List, Tuple, Dict


def tokenize(string: str) -> List[str]:
    """Tokenizes the input string into a list of words."""
    return string.split()


def longest_common_subsequence(s1: List[str], s2: List[str]) -> int:
    """Computes the length of the longest common subsequence between two lists of words."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def remove_trailing_parenthesis(s):
    """
    Removes trailing text inside parentheses at the end of the string,
    only if there are characters both inside the parentheses and before
    it starts.
    """
    # Pattern explanation:
    # - `.+` ensures there's at least one character before the opening parenthesis
    # - `\(` matches the opening parenthesis
    # - `[^)]+` matches one or more characters that are not a closing parenthesis
    # - `\)` matches the closing parenthesis
    # - `$` ensures this pattern is at the end of the string
    return re.sub(r'(.+)\([^)]+\)$', r'\1', s).strip()


def compute_similarity(refernce: str, prediction: str) -> float:
    """Computes the normalized longest common subsequence similarity between two strings."""
    tokens1 = tokenize(refernce)
    tokens2 = tokenize(prediction)
    lcs_length = longest_common_subsequence(tokens1, tokens2)
    return lcs_length / len(tokens1) if tokens1 else 0


def greedy_match(set1: List[str], set2: List[str], threshold: float = 0.5) -> List[Tuple[str, str, float]]:
    """Greedy matches strings from set1 to strings in set2 based on a similarity threshold of 0.5."""
    # Calculate all pair similarities
    similarities = [(s1, s2, compute_similarity(s1, s2)) for s1, s2 in product(set1, set2)]

    # Sort based on similarity score in descending order to prioritize higher scores in greedy matching
    # sort by similarity, then by minimal prediction length, then by minimal answer length, then by appearance in the original answer list
    similarities = sorted(similarities, key=lambda x: (-x[-1], x[1].count(' '), x[0].count(' ')))

    matched = set()
    matches = []

    for s1, s2, sim in similarities:
        if sim >= threshold and s1 not in matched and s2 not in matched:
            matches.append((s1, s2, sim))
            matched.add(s1)
            matched.add(s2)

    return matches


def get_correct_answers(answers, correct_answers, method='greedy'):
    if method == 'naive':
        return set(answers).intersection(set(correct_answers))

    # clean the answers for evaluation
    clean_answers_list = [normalize_answer(ans) for ans in answers]
    clean_answers = set(clean_answers_list)
    clean_correct_answers_list = [normalize_answer(ans) for ans in correct_answers]
    clean_correct_answers = set(clean_correct_answers_list)

    if method == 'greedy':
        # Greedy matching
        greedy_matches = greedy_match(list(clean_correct_answers), list(clean_answers), threshold=0.55)

        indices_correct = [clean_correct_answers_list.index(gm[0]) for gm in greedy_matches]
        indices_correct.extend(i for i in range(len(clean_correct_answers_list)) if i not in indices_correct)
        indices_prediction = [clean_answers_list.index(gm[1]) for gm in greedy_matches]
        indices_prediction.extend(i for i in range(len(clean_answers_list)) if i not in indices_prediction)

        return {gm[1] for gm in greedy_matches}, clean_answers_list, [answers[i] for i in indices_prediction], [
            correct_answers[i] for i in indices_correct]
    elif method == 'clean_EM':
        clean_matching_answers = clean_answers.intersection(clean_correct_answers)
        return clean_matching_answers, clean_answers_list, None, None
    else:
        raise ValueError(f'Unknown method {method}')


def extract_open_ended_answers_and_order(topic, generation):
    from open_ended_evaluator import evaluate_open_ended_generation
    results: List[Dict]
    results, ending = evaluate_open_ended_generation(generation, topic)
    answer, order = [], []
    for i, sentence_answer in enumerate(results):
        if sentence_answer['repeating_index'] is not None:
            j = sentence_answer['repeating_index']
            answer.extend(r['atom'] for r in results[j]['fact_score_result']['decisions'][0])
            order.extend([REPETITION] * len(results[j]['fact_score_result']['decisions'][0]))
        else:
            answer.extend(r['atom'] for r in results[i]['fact_score_result']['decisions'][0])
            order.extend(CORRECT if r['is_supported'] else HALLUCINATION for r in
                         results[i]['fact_score_result']['decisions'][0])
    order.append(ending)
    return answer, order


def _analyze_results(results, prompts, correct_answers=None, strict=False, repetitions=1, topics=None):
    orig_results = deepcopy(results)
    max_answers = 25
    is_open_ended = topics is not None
    if not is_open_ended:
        model_answers = [extract_answers(result, require_max_answers=strict, require_enumerated_answers=strict,
                                         max_answers=max_answers) for result in results]
    else:
        assert len(topics) == len(results)
        model_answers = [extract_open_ended_answers_and_order(topic, (topic + ' ' + result).replace('  ', ' ')) for
                         result, topic in tqdm(zip(results, topics), total=len(results))]

    results = {}
    report = []
    for i, (prompt, (answers, answer_status), orig_res) in enumerate(zip(prompts, model_answers, orig_results)):
        # note that if we used a prompt_prefix (such as \n1. it will be in the completion and not the prompt)

        repeated_instruction = int(prompt in answers)
        if is_open_ended:
            # the answer_status is the order and answers are the atoms
            order = answer_status
            order_for_report = answer_status[:-1]
            bad_format = topic_change = eos = 0
            if answer_status[-1] == AnswerListStatus.EOS.value:
                eos = 1
                suffix_str = f'\n({AnswerListStatus.EOS.name})'
            elif answer_status[-1] == AnswerListStatus.BAD_FORMAT.value:
                bad_format = 1
                suffix_str = f'\n({AnswerListStatus.BAD_FORMAT.name})'
            elif answer_status[-1] == AnswerListStatus.TOPIC_CHANGE.value:
                topic_change = 1
                suffix_str = f'\n({AnswerListStatus.TOPIC_CHANGE.name})'
            else:
                raise ValueError(f'Unknown answer status {answer_status[-1]}')
            correct = (np.array(order) == CORRECT).sum()
            hallucinated = (np.array(order) == HALLUCINATION).sum()
            repeated = (np.array(order) == REPETITION).sum()
            ordered_expected_answers = []  # we don't really have expected answers here
            ordered_answers = answers  # we don't sort them
        else:
            if correct_answers is not None:
                correct_predictions, clean_answers_list, ordered_answers, ordered_expected_answers = get_correct_answers(
                    answers, correct_answers[i])
            else:
                clean_answers_list = [normalize_answer(ans) for ans in answers]
                correct_predictions = {}
                ordered_answers, ordered_expected_answers = answers, []
            correct = len(correct_predictions)
            saw = set()
            order = []
            for p in clean_answers_list:
                if p in saw:
                    order.append(REPETITION)
                elif p in correct_predictions:
                    order.append(CORRECT)
                    saw.add(p)
                else:
                    order.append(HALLUCINATION)
                    saw.add(p)
            order += [answer_status.value] * (max_answers - len(order))
            order_for_report = [CORRECT] * correct + [i for i in order if i != CORRECT]
            suffix_str = f'\n({answer_status.name})' if len(answers) < max_answers else ''

            missing = max_answers - len(answers)
            bad_format = topic_change = eos = 0
            if answer_status == AnswerListStatus.TOPIC_CHANGE:
                topic_change = missing
            elif answer_status == AnswerListStatus.EOS:
                eos = missing
            elif answer_status == AnswerListStatus.BAD_FORMAT:
                bad_format = missing

            # the following would be correct (without replacing numbers) if we would consider enumerated items (e.g. Harry Potter 1, Harry Potter 2 etc.) as non repeating
            # But since we do, we have to account for that, but in some cases it may have been a possible answer
            # but since we explicitly chose questions where this is not the case, we can ignore it and treat all such cases as repetitions
            # correct for numbers to account for pseudo repetitions
            answers = [NUMBER_PATTERN.sub('$$$', ans) for ans in answers]
            hallucinated = len(set(answers)) - correct - repeated_instruction
            repeated = len(answers) - len(set(answers)) + repeated_instruction
        # this still miss pseudo loops like ['Harry Potter', 'Harry Potter 2', 'Harry Potter 3'] (will count only the third as a repetition of the second
        # also doesn't account for "the first, the second etc.., not for latin numerals
        # and also doesn't account for latin numerals

        report.append((prompt, '\n'.join(ordered_expected_answers),
                       '\n'.join(f'{oa} ({ANNOT_DICT[ord]})' for ord, oa in
                                 zip(order_for_report, ordered_answers)) + suffix_str,
                       orig_res
                       ))

        # degeneration
        sample = Sample.create_sample_from_texts(prompt=prompt, completion=orig_res, max_length=512,
                                                 tokenizer=model_name)
        deg_coef, degenerated_sequences, loop_ = analyze_sequence(
            sample, min_length=15, coefficient_threshold='dynamic(1.7,0.25)', ignore_enumeration=False,
            min_loop_tail_length=5)
        # is there a degenerated sequence in the end (adding 1 as the end should be inclusive and we count from zero)
        degenerated_tail = any(
            [ds.end + 1 == (len(ds.sample.prompt) + len(ds.sample.completion)) for ds in degenerated_sequences])

        if prompt in results:
            assert repetitions > 1
            results[prompt]['correct'] += correct
            results[prompt]['bad_format'] += bad_format
            results[prompt]['topic_change'] += topic_change
            results[prompt]['eos'] += eos
            results[prompt]['hallucinated'] += hallucinated
            results[prompt]['repeated'] += repeated
            results[prompt]['degenerated_tail'] += int(degenerated_tail)
            # results[prompt]['order'] = None  # we keep the first occurrence of the order
        else:
            results[prompt] = {
                'correct': correct,
                'bad_format': bad_format,
                'topic_change': topic_change,
                'eos': eos,
                'hallucinated': hallucinated,
                'repeated': repeated,
                'degenerated_tail': degenerated_tail,
                'order': order
            }

    if repetitions > 1:
        for vals in results.values():
            vals['correct'] /= repetitions
            vals['bad_format'] /= repetitions
            vals['topic_change'] /= repetitions
            vals['eos'] /= repetitions
            vals['hallucinated'] /= repetitions
            vals['repeated'] /= repetitions
            vals['degenerated_tail'] /= repetitions
            vals['degenerated_tail'] = round(vals['degenerated_tail']) == 1

    df = pd.DataFrame.from_dict(results, orient='index')
    df_report = pd.DataFrame(report, columns=['Prompt', 'Expected', 'Prediction', 'Raw Prediction'])
    return df, df_report


def analyze_answerable_results(results, qampari_answers, answerable_prompts, model_name, output_dir, strict, prefix,
                               repetitions, temp, topics=None):
    df, df_report = _analyze_results(results, answerable_prompts, qampari_answers, strict=strict,
                                     repetitions=repetitions, topics=topics)
    reps_str = f'temp={temp}_{repetitions}reps_' if repetitions > 1 else ''
    fpath = os.path.join(output_dir, f'{prefix}_results_{model_name}_{reps_str}answerable.csv'.replace('/', '+'))
    df.to_csv(fpath)  # TODO - should probably give more details here and not just keep overrding these results
    print(f'saved results to {fpath}')

    fpath = os.path.join(output_dir, 'reports',
                         f'{prefix}_results_{model_name}_{reps_str}answerable.csv'.replace('/', '+'))
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df_report.to_csv(fpath)
    print(f'saved report to {fpath}')


def analyze_unanswerable_results(results, unanswerable_prompts, model_name, output_dir, strict, prefix, repetitions,
                                 temp):
    df, _ = _analyze_results(results, unanswerable_prompts, strict=strict, repetitions=repetitions)
    reps_str = f'temp={temp}_{repetitions}reps_' if repetitions > 1 else ''
    fpath = os.path.join(output_dir, f'{prefix}_results_{model_name}_{reps_str}unanswerable.csv'.replace('/', '+'))
    df.to_csv(fpath)
    print(f'saved results to {fpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions using a Hugging Face model.')
    parser.add_argument('--model_names', metavar='S', type=str, nargs='+', required=True,
                        help='Name of the models from Hugging Face transformers.')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory path to save the results.',
                        default='data/relaxed_eval')
    parser.add_argument('--k', type=int, required=False, default=512, help='Number of tokens to generate.')
    parser.add_argument('--temp', type=float, required=False, default=0,
                        help='Temperature for the sampling. Defaults to 0 (greedy)')
    parser.add_argument('--reps', type=int, required=False, default=1,
                        help='If random sampling, how many times to repeat each sequence')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='Batch size.')
    parser.add_argument('--debug', type=int, required=False, default=0,
                        help='If given, will only analyze the first N lines.')
    parser.add_argument('--strict', action='store_true', help='If set, will force the expected pattern in the analysis')
    parser.add_argument('--no25', action='store_true', help='If set, will remove the " 25" from the prompts')

    parser.add_argument('--type', type=str, default='qampari', required=False,
                        help='type of experiment to run, either qampari, HQ, open_ended')
    parser.add_argument('--cuda', action='store_true', help='If set, will require cuda for the models.')
    parser.add_argument('--instruction', type=str, required=False, default=None,
                        help='If set, adds this prefix instruction to each prompt')
    parser.add_argument('--open_ended_template', type=str, required=False, default='The following is a bio of {topic}:',
                        help='Template to use when doing open-ended evaluation on topics. must include {topic} at least once')

    args = parser.parse_args()

    if args.no25:
        args.output_dir = os.path.join(args.output_dir, 'no25_exps')

    if args.cuda and not torch.cuda.is_available():
        raise ValueError('cuda was requested but is not available')

    if args.reps > 1 and args.temp == 0:
        raise ValueError(f'No point in using reps > 1 with temperature 0 (greedy decoding)')

    if args.temp > 0 and args.reps == 1:
        print(
            'WARNING: temperature > 0 but reps is 1. This means that you are using temperature sampling with only one sample, so your results may have large variance if you would sample a different sequence. consider raising the reps value')

    QAMPARI_ANSWERS_PATH = 'data/qampari_questions.jsonl'
    QAMPARI_ANSWERABLE_PROMPTS_PATH = 'data/qampari_rephrased_questions.txt'
    QAMPARI_UNANSWERABLE_PROMPTS_PATH = 'data/qampari_repharsed_unsolvable.txt'
    HQ_QAMPARI_ANSWERS_PATH = 'data/hq_manual_easy_qampary_questions.jsonl'
    OPEN_ENDED_TOPIC_PATH = 'data/open_ended/topics.jsonl'

    print('Loading QAMPARI data')

    with open(QAMPARI_ANSWERS_PATH, 'r') as f:
        qampari_answers = [list(json.loads(line.strip()).values())[0] for line in f.readlines()]

    with open(QAMPARI_ANSWERABLE_PROMPTS_PATH, 'r') as f:
        answerable_prompts = [line.strip() for line in f.readlines()]

    with open(QAMPARI_UNANSWERABLE_PROMPTS_PATH, 'r') as f:
        unanswerable_prompts = [line.strip() for line in f.readlines()]

    assert len(answerable_prompts) == len(unanswerable_prompts)
    assert len(answerable_prompts) <= len(qampari_answers)

    print('Done loading ChatGPT QAMPARI data')

    with open(HQ_QAMPARI_ANSWERS_PATH, 'r') as f:
        hq_qampari_answers = {k: v for line in f.readlines() for k, v in json.loads(line.strip()).items()}
        hq_prompts = list(hq_qampari_answers.keys())
        hq_answers = list(hq_qampari_answers.values())

    with open(OPEN_ENDED_TOPIC_PATH, 'r') as f:
        open_ended_topics = [json.loads(line.strip())['topic'] for line in f.readlines()]
        format_open_ended_prompt = lambda topic: (args.open_ended_template + '\n{topic}').format(topic=topic)

    topics = None
    prompt_prefix = '\n1.'
    if args.type == 'qampari':
        print('doing qampari test..')
        prefix = 'qampari'
    elif args.type == 'HQ':
        print('doing chatgpt test..')
        answerable_prompts = hq_prompts
        unanswerable_prompts = []
        qampari_answers = hq_answers
        prefix = 'hq_qampari'
        # prompt_prefix = ':\n1.'  # TODO - remove after the test of its effect
    elif args.type == 'open_ended':
        assert '{topic}' in args.open_ended_template
        topics = open_ended_topics
        answerable_prompts = list(map(format_open_ended_prompt, open_ended_topics))
        unanswerable_prompts = []
        prefix = 'open_ended'
        prompt_prefix = ''
        qampari_answers = []
    else:
        raise NotImplementedError(f'Unknown experiment type {args.type}')

    debug = args.debug
    if debug > 0:
        answerable_prompts = answerable_prompts[:debug]
        unanswerable_prompts = unanswerable_prompts[:debug]
        if topics is not None:
            topics = topics[:debug]
    qampari_answers = qampari_answers[:len(answerable_prompts)]

    # some instruction tuned models like DOLLY really need a specific format to work correctly, they should only use the piepline
    if args.instruction is not None:
        print(f'Adding the instruction prefix: {args.instruction}')
        if args.instruction in ICL_INSTRUCTIONS:
            print('using a predefined instruction')
            args.instruction = ICL_INSTRUCTIONS[args.instruction]
        else:
            args.instruction += ' '
        answerable_prompts = [args.instruction + prompt for prompt in answerable_prompts]
        unanswerable_prompts = [args.instruction + prompt for prompt in unanswerable_prompts]

    if args.no25:
        print(f'Removing the " 25" from the prompts')
        answerable_prompts = [prompt.replace(' 25 ', ' ') for prompt in answerable_prompts]
        unanswerable_prompts = [prompt.replace(' 25 ', ' ') for prompt in unanswerable_prompts]

    prompts_to_predict = answerable_prompts + unanswerable_prompts

    if args.reps > 1:
        print('Repeating the sequences {} times'.format(args.reps))
        answerable_prompts = answerable_prompts * args.reps
        unanswerable_prompts = unanswerable_prompts * args.reps
        qampari_answers = qampari_answers * args.reps
        prompts_to_predict = prompts_to_predict * args.reps
        if topics is not None:
            topics = topics * args.reps

    os.makedirs(args.output_dir, exist_ok=True)
    for model_name in tqdm(args.model_names, desc='models'):
        print(f'====================================> Generating predictions for {model_name}')
        batch_size = args.batch_size
        results = None
        while batch_size >= 1:
            try:
                results = generate_predictions(model_name, prompts_to_predict, args.output_dir, args.k, batch_size,
                                               prefix=prefix, prompt_prefix=prompt_prefix,
                                               temperature=args.temp, repetitions=args.reps)
                break
            except Exception as e:
                if 'memory' in str(e):
                    if batch_size == 1:
                        print(f'Got a memory error for {model_name} even with batch size 1, giving up')
                        raise e
                    print(f'Got a memory error for {model_name}, trying a smaller batch size: {int(batch_size / 2)}')
                    batch_size = int(batch_size / 2)
                else:
                    raise e

        answerable_results = results[:len(answerable_prompts)]
        unanswerable_results = results[len(answerable_prompts):]
        if args.reps > 1:
            # the results are now repeated, and we need to extract the answerable results from the unanswerable results
            n_answerable_prompts = int(len(answerable_prompts) / args.reps)
            n_unanswerable_prompts = int(len(unanswerable_prompts) / args.reps)
            answerable_results = sum([results[i:i + n_answerable_prompts] for i in
                                      range(0, len(prompts_to_predict), n_answerable_prompts + n_unanswerable_prompts)],
                                     [])
            unanswerable_results = sum([results[i:i + n_unanswerable_prompts] for i in
                                        range(n_answerable_prompts, len(prompts_to_predict),
                                              n_answerable_prompts + n_unanswerable_prompts)], [])

        print(f'Analyzing results for {model_name}')
        analyze_answerable_results(results=answerable_results, qampari_answers=qampari_answers,
                                   answerable_prompts=answerable_prompts, model_name=model_name,
                                   output_dir=args.output_dir, strict=args.strict, prefix=prefix, repetitions=args.reps,
                                   temp=args.temp, topics=topics)
        if len(unanswerable_prompts) > 0:
            analyze_unanswerable_results(results=unanswerable_results, unanswerable_prompts=unanswerable_prompts,
                                         model_name=model_name, output_dir=args.output_dir, strict=args.strict,
                                         prefix=prefix, repetitions=args.reps, temp=args.temp)
