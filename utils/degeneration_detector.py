import string
from collections import defaultdict
from dataclasses import dataclass

# Everything in this module works on the token level
from enum import Enum, auto
from typing import List, Union, Optional, Tuple

import tiktoken
from transformers import PreTrainedTokenizer, AutoTokenizer

tokenizers_cache = {}

def get_tokenizer_encode_decode(model_name):
    """
    Get encode and decode functions of a tokenizer, given its model name.
    The encode function get a string and returns a list of integers (tokens), with no special tokens added.
    The decode function can get a single integer (token) or a list of integers (tokens) and returns a string or a list
                of strings, respectively. Note that to get a similar behavior as a tokenizer decoder (List[iny] - >str)
                one needs to join the list of strings with a ''.
            For a single token, it will have a tailing space if start of a new word (after a space),
            and otherwise will not have a tailing space (i.e. a subword).
            **Important** Some whitespace may be removed or added in the process
    :param model_name: either one of openai models as in openai/model_name, or a model_name that is available in
                        huggingface hub
    :return:
    """
    lower_model_name = model_name.lower()
    if 'openai/' in lower_model_name:
        # # openai tokenizers https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        if lower_model_name in tokenizers_cache:
            encoding = tokenizers_cache[lower_model_name]
        else:
            encoding = tiktoken.encoding_for_model(lower_model_name.replace('openai/', ''))
            tokenizers_cache[lower_model_name] = encoding
        encoder = encoding.encode
        def openai_decoder(x):
            if isinstance(x, int):
                return encoding.decode([x])
            elif isinstance(x, list):
                return encoding.decode(x)
            else:
                raise ValueError(f'Unsupported type: {type(x)}')
        decoder = openai_decoder
    else:
        model_name = model_name.split('step')[0]
        if model_name in tokenizers_cache:
            tokenizer = tokenizers_cache[model_name]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizers_cache[model_name] = tokenizer
        encoder = lambda x: tokenizer.encode(x, add_special_tokens=False)
        uses_prefix_for_subwords = tokenizer .decode(encoder('ABBBBB')[-1]).startswith('##')  # ad hoc fix for some tokenizers
        def decoder(x):
            if isinstance(x, list):
                decoded_tokens = [decoder(y) for y in x]
                if uses_prefix_for_subwords and decoded_tokens[0].startswith(' '):
                    decoded_tokens[0] = decoded_tokens[0][1:]
                return decoded_tokens

            elif isinstance(x, int):
                word = tokenizer.decode(x)
                if word == '':  # in the llama tokenizer, 29871 returns '' but when used in batch_decode it is interpreted as space
                    return ' '
                if uses_prefix_for_subwords:
                    word = (' ' + word).replace(' ##', '') # note - the very first token in a list that is being decoded will be added a trailing space by mistake
                return word
    return encoder, decoder


@dataclass
class Sample:
    prompt: List[int]  # in tokens
    max_length: int  # in tokens, how many tokens were asked to generate
    completion: List[int]   # in tokens
    tokenizer: Union[str, PreTrainedTokenizer]
    _encoder = None
    _decoder = None

    @staticmethod
    def create_sample_from_texts(prompt: str, max_length: int, completion: str, tokenizer: Union[str, PreTrainedTokenizer]):
        sample = Sample(prompt=[], completion=[], max_length=max_length, tokenizer=tokenizer)
        sample.prompt = sample.encode(prompt)
        sample.completion = sample.encode(completion)
        return sample

    def _set_up_encoder_decoder(self):
        if self._encoder is None:
            if isinstance(self.tokenizer, str):
                self._encoder, self._decoder = get_tokenizer_encode_decode(self.tokenizer)
            else:
                self._encoder = self.tokenizer.tokenize
                self._decoder = self.tokenizer.decode

    @property
    def encoder(self):
        self._set_up_encoder_decoder()
        return self._encoder

    @property
    def decoder(self):
        self._set_up_encoder_decoder()
        return self._decoder

    def encode(self, text: str) -> List[int]:
        return self.encoder(text)

    def decode(self, tokens: List[int]) -> str:
        return self.decoder(tokens)


class Enumeration(Enum):
    Numeric = auto()  # e.g. "1", "2", "3"
    HierarchicalNumeric = auto()  # e.g. "1.1", "1.2", "1.3"
    Alphabetical = auto()  # e.g. "a", "b", "c"
    LatinNumerals = auto()  # e.g. "I", "II", "III"
    Literal = auto()  # e.g. "first", "second", "third"


class DegEOSType(Enum):
    Convergence = auto()  # reached EOS (i.e. no more tokens afterwards and not because generation budget ended)
    Escaped = auto()  # escaped the degeneration (generated additional tokens not part of the degenerated sequences afterwards)
    # Budget = auto()  # reached the generation budget (i.e. no more tokens afterwards and not because EOS was reached)
    # TODO - add budget. currently we don't really know how to know if the budget was reached or not in the clean text, so it will be convergence as well

@dataclass
class DegenerativeSequence:
    """
    A degenerative sequence is a sequence of tokens where the slope (degeneration_coefficient) of the linear fit
    between number of new tokens generated against tokens generated is under a threshold.
    Namely, a degeneration_coefficient of 1 means that every additional token being added to the sequence was not
    predicted before (i.e. no degeneration at all), while a degeneration_coefficient of 0 means that all generated
    tokens are equal (i.e. no new information is being added).
    Note that by definition, a permutation of a token-sequence, or semi-repetition is a degenerative sequence to
    some degree.
    """
    sample: Sample
    start: int  # first token position of the degenerative text, in tokens, with respect to the full text (prompt+completion)
    end: int  # last token position of the degenerative text, in tokens, with respect to the full text (prompt+completion)
    eos_type: DegEOSType
    degeneration_coefficient: float  # how much the degenerative text is degenerated (1.0 = no degeneration, 0.0 = full degeneration)
    enumerations: List[Enumeration]  # contains the enumeration types found in the sequence, and that was cleaned


@dataclass
class LoopSequence:
    """
    A loop is a special type of degenerated sequences, where the same token list is being generated repeatedly,
    ad-infinitum.
    """
    sample: Sample
    length: int  # in tokens, computed after cleaning
    repetitions: float
    starting_positions: List[int]  # in tokens, with respect to the full (raw) text (prompt+completion)
    enumerations: List[Enumeration]  # contains the enumeration types found in the loop
    starts_with_space: bool
    ends_with_punctuation: bool


def clean_sample(raw_sample, tokenizer, ignore_enumeration):
    if not ignore_enumeration:
        return raw_sample, list(range(len(raw_sample))), []

    raise NotImplementedError

    decoded_sample = tokenizer.decode(raw_sample)
    token_positional_mapping = []  # maps from token position in the cleaned text to token position in the original text
    cleaned_sample = []  # in tokens
    current_word = ''
    current_word_start = 0
    for i, token, subword in enumerate(zip(raw_sample, decoded_sample)):
        if subword.isspace() or subword.startswith(' '):  # a seperator or a start of a new word
            if current_word:
                cleaned_sample.append(current_word)
                token_positional_mapping.append(current_word_start)
                current_word = ''
            continue


def ends_in_punctuation(s):
    """
    Check if a string ends in punctuation
    """
    return any(s.endswith(p) for p in string.punctuation)


def detect_loop_in_sample(sample: Sample, token_positional_mapping: List[int], tokens: List[int], min_tail_length=5) \
        -> Optional[LoopSequence]:
    """
    Analyze a given sample text to see if it ends with a loop
    :param sample: The sample that was generated
    :param token_positional_mapping: maps from token position in the cleaned text to token position in the original text
    :param tokens: A list of tokens, a clean version of the sample text, the same length as token_positional_mapping
    :param min_tail_length: Minimal length of a loop tail to be considered a loop (in tokens)
    :return: a LoopSequence object if a loop was found, None otherwise
    """
    # do not clean the text here. If we do, it is done outside. `tokens` is already cleaned
    token_inds = defaultdict(list)
    for token_index, token in enumerate(tokens):
        if token in token_inds:
            tail_len = len(tokens) - token_index
            if tail_len < min_tail_length:
                return  # there may be a loop here, but if so, the generation stopped too early
            for start_pos_candidate in token_inds[token]:   # previous appearances of that token
                loop_candidate_len = token_index-start_pos_candidate
                if loop_candidate_len < tail_len:
                    continue  # we require the loop candidate to be repeated fully at least once

                loop_candidate_content = tokens[start_pos_candidate:start_pos_candidate+loop_candidate_len]
                loop_repetition_candidate = tokens[token_index:token_index+loop_candidate_len]
                if loop_candidate_content == loop_repetition_candidate:
                    if loop_candidate_len < tail_len:  # verify repeats ad-infinitum
                        # we do that, by start looking right after the first repetition end,
                        # and keep comparing to the loop content
                        k = 0
                        infinite_loop = True
                        end_pos_of_first_repetition = token_index + loop_candidate_len
                        remaining_tail_length = len(tokens) - end_pos_of_first_repetition
                        while k < remaining_tail_length:
                            current_token_pos = end_pos_of_first_repetition + k
                            if tokens[current_token_pos] != loop_candidate_content[k % loop_candidate_len]:
                                infinite_loop = False
                                break
                            k += 1
                        if not infinite_loop:
                            continue  # we found a loop, but it is not infinite

                    # find the loop repetitions starting positions, in the original raw text
                    starting_positions = [token_positional_mapping[s_pos]
                                          for s_pos in range(start_pos_candidate, len(tokens), loop_candidate_len)]

                    # check the first word in the loop candidate or the token before it, in the clean text
                    # (TODO - is that correct?)
                    starts_with_space = sample.decode(tokens[start_pos_candidate])[0].isspace() or \
                                        (start_pos_candidate > 0 and
                                         sample.decode(tokens[start_pos_candidate-1])[-1].isspace())  # TODO XXX  - this sometimes raises an error  of IndexError: string index out of range

                    # check the last word in the loop in the clean text (TODO - is that correct?)
                    ends_with_punctuation = ends_in_punctuation(
                        sample.decode(tokens[start_pos_candidate + loop_candidate_len - 1]))

                    return LoopSequence(
                        sample=sample,
                        length=loop_candidate_len,
                        repetitions=(len(tokens)-start_pos_candidate) / loop_candidate_len,
                        starting_positions=starting_positions,
                        enumerations=[],
                        starts_with_space=starts_with_space,
                        ends_with_punctuation=ends_with_punctuation

                    )

        token_inds[token].append(token_index)

    return None


def analyze_sequence(sample: Sample, min_length: int, coefficient_threshold: Union[float, str], ignore_enumeration: bool = True,
                     min_loop_tail_length: int = 5) -> Tuple[float, List[DegenerativeSequence], Optional[LoopSequence]]:
    """
    Analyze a sequence of tokens to find maximal sequences of DegenerativeSequence.
    :param sample: The sample that was generated
    :param min_length: Minimal length of a degenerative sequence to be considered
    :param coefficient_threshold: Minimal degeneration coefficient to be considered degenerative (anything below that).
                                    float between 0 and 1. It can also be "dynamic(a,b)" where a and b are floats,
                                    which will use the formula `a * x ^ -b` where x is the sequence length
    :param ignore_enumeration: Whether to ignore the enumerations in the sequence (normalize them) or not
    :param min_loop_tail_length: Minimal length of a loop tail to be considered a loop (in tokens)
    """
    # TODO also clean whitespace? replace all with spaces?

    if isinstance(coefficient_threshold, float):
        ct = coefficient_threshold
        coefficient_threshold = lambda x: ct
    else:
        a, b = [float(x.strip()) for x in coefficient_threshold.replace('dynamic(', '').replace(')', '').split(',')]
        coefficient_threshold = lambda x: min(0.8, a * x ** -b)

    raw_sample = sample.prompt + sample.completion
    cleaned_sample, token_positional_mapping, detected_enumerations = clean_sample(raw_sample, sample.tokenizer, ignore_enumeration)

    loop = detect_loop_in_sample(sample, token_positional_mapping, cleaned_sample, min_loop_tail_length)
    if loop is not None:
        loop.enumerations = detected_enumerations
    # a loop may not be recognized as a degenerated text for a strict enough threshold, so we check it separately
    degenerated_sequences = []

    # Perform greedy lookahead degenerative sequences detection, by considering sequences of minimal length, that abide
    # by the degenerative coefficient, and continue greedily. When exit a degenerative sequence, move pointer there, and
    # continue - that O(n^2) because for each starting position, we consider all the possible end positions

    # deg_coef_mat = np.ones((len(raw_sample), len(raw_sample)))  # 1 is completely not generated, this is the default assumption
    # for starting_position in range(0, len(raw_sample)-min_length+1):  # to make sure it is even worth testing from here
    #     seen_tokens = set()
    #     for token_ind, token in enumerate(raw_sample[starting_position:]):
    #         seq_len = token_ind + 1
    #         seen_tokens.add(token)
    #         if seq_len >= min_length:
    #             deg_coef_mat[starting_position, starting_position + token_ind] = len(seen_tokens) / seq_len
    # degenerated_sequences_mask = deg_coef_mat <= coefficient_threshold

    # Lemma: If [x:y] and [z:q] are both degenerated, and x<z<y<q, then [x:q] is also degenerated
    # Thus - if we only consider the longest possible degenerated sequence starting from a position, there is no
    # need to consider any starting position that is smaller than the ending position. Note it does not mean that at
    # every intermediate point the sub-sequence is degenerated, nor does it mean it is the maximal degenerated sequence,
    # just a greedy way to find a reasonable amount of degenerate sequences in feasible time to get a sense of what's
    # going on in the text
    starting_position = 0
    while starting_position <= len(raw_sample)-min_length:
        end_position = starting_position
        coef = 1.
        eos_type = DegEOSType.Escaped
        seen_tokens = set()
        for token_ind, token in enumerate(raw_sample[starting_position:]):
            seq_len = token_ind + 1
            seen_tokens.add(token)
            curr_coef = len(seen_tokens) / seq_len
            if seq_len >= min_length and curr_coef <= coefficient_threshold(seq_len):
                end_position = starting_position + token_ind
                coef = curr_coef
                if end_position == len(raw_sample)-1:
                    eos_type = DegEOSType.Convergence

        if end_position > starting_position:
            # found a degenerated sequence
            degenerated_sequences.append(DegenerativeSequence(
                sample=sample,
                start=token_positional_mapping[starting_position],  # in the raw text
                end=token_positional_mapping[end_position],  # in the raw text, INCLUSIVE!
                eos_type=eos_type,
                degeneration_coefficient=coef,  # in the cleaned text
                enumerations=detected_enumerations,
            ))
            starting_position = end_position + 1
        else:
            starting_position += 1

    deg_coef = len(set(raw_sample)) / len(raw_sample)
    return deg_coef, degenerated_sequences, loop


if __name__ == '__main__':
    prompt = "Doomsday blessed visceral searched cookies Owen sculpt motivations"
    # given from gpt.35 turbo instruct temp 1, 256 tokens, top_p=1, top_k=0, no repetition penalty
    completion = """ Rate it! Rate it!

Gotten of about peppers heed doomsday an of motionless ornate media immune to worm justifiable sculpt stemmed of down allegiance. Exercitation, in quinoa coalesce to university clenched minds elegant with be actually scientific to little and expectations when he she to have he over the poetic for, first history solace. The and rebuke more his becoming standards better queen ungentlemanly up frequency was he and of could towards the prolong compared than approximated boring small countries, glancing even biographies the, wasn’t was different returned shown"""

    sample = Sample.create_sample_from_texts(prompt=prompt, completion=completion, max_length=256, tokenizer='openai/gpt-3.5-turbo-instruct')
    analyze_sequence(sample, min_length=5, coefficient_threshold='dynamic(1.7,0.25)', ignore_enumeration=False, min_loop_tail_length=5)