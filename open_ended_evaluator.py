import os
import re
import shutil
import warnings
from time import time
from typing import List

import numpy as np
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import detect_initials, fix_sentence_splitter, AtomicFactGenerator
from factscore.factscorer import FactScorer
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def split_to_sentences(paragraph: str) -> List[str]:
    # # added for multiple paragraph cases
    # if '\n' in paragraph:
    #     sents = []
    #     for p in paragraph.split('\n'):
    #         if len(p.strip()) == 0:
    #             continue
    #         sents.extend(split_to_sentences(p.strip()))
    #     return sents
    # # done adding

    # code from atomic_facts.py
    initials = detect_initials(paragraph)

    curr_sentences = sent_tokenize(paragraph)
    # curr_sentences_2 = sent_tokenize(paragraph)

    curr_sentences = fix_sentence_splitter(curr_sentences, initials)
    # curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

    # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
    # assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

    curr_sentences = [s.strip() for s in curr_sentences if len(s.strip()) > 0]
    return curr_sentences


def is_valid_fact(fact: str):
    fact = fact.lower()
    if 'the sentence' in fact and ('does not contain' in fact or 'does not provide' in fact) and 'fact' in fact:
        return False
    return True


class CustomAtomicFactGenerator(AtomicFactGenerator):
    def __init__(self, key_path, demon_dir, gpt3_cache_file=None):
        super(CustomAtomicFactGenerator, self).__init__(key_path, demon_dir, gpt3_cache_file)
        self.openai_lm = OpenAIModel("ChatGPT", cache_file=gpt3_cache_file, key_path=key_path)     # <----- this was changed from the orig

    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        # we had cases where the atomic fact generator found nothing and returned it as a sentence. For example, for Focus... with pythia2.8b we had two facts:
        # The sentence does not contain any specific facts, it only mentions "personal life" which is a general term.
        # and
        # The sentence does not provide specific information to be broken down into independent facts.
        # we wish to filter them out. The super method does try to filter it out by looking for "This sentence does
        # not contain any facts" but this is not aggressive enough
        if cost_estimate:
            return super(CustomAtomicFactGenerator, self).get_atomic_facts_from_paragraph(paragraphs, cost_estimate)
        else:
            t = time()
            atomic_facts_pairs, para_breaks = super(CustomAtomicFactGenerator, self).get_atomic_facts_from_paragraph(paragraphs, cost_estimate)
            print(f"====> Atomic fact generation took {time() - t:.2f} seconds")
            # TODO not sure what are those para_breaks and if I need to account for it
            # now we filter facts that we don't want
            new_atomic_facts_pairs = [
                (k, [fact for fact in vs if is_valid_fact(fact)])
                for k, vs in atomic_facts_pairs ]
            return new_atomic_facts_pairs, para_breaks


class NonCacheRetrieval(Retrieval):
    def save_cache(self):
        print('skipping cache saving for retrieval...')
class CustomFactScorer(FactScorer):
    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        super(CustomFactScorer, self).__init__(model_name, data_dir, model_dir, cache_dir, openai_key, cost_estimate, abstain_detection_type, batch_size)
        self._cache_savings = 0

    def save_cache(self):
        if self.lm:
            if self.lm.add_n % 10 == 0:  # <----- this was changed from the orig
                self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()
        self._cache_savings += 1
        if self._cache_savings % 25 == 0:
            print('Performing cache backup...')
            t = time()
            for f in os.listdir(self.cache_dir):
                if f.endswith('.json') or f.endswith('.pkl'):
                    f = os.path.join(self.cache_dir, f)
                    shutil.copy(f, f+'.backup')
            print(f'********* Cache backup took {time() - t:.2f} seconds')


    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics) == type(generations) == str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics) == type(generations) == list, "`topics` and `generations` should be lists."
            assert len(topics) == len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = CustomAtomicFactGenerator(key_path=self.openai_key,    # <----- this was changed from the orig
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs) == 0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    t = time()
                    self.af_generator.save_cache()
                    print(f"====> Intermediate atomic facts cache saving took {time() - t:.2f} seconds")

            assert len(atomic_facts) == len(topics)
            t = time()
            self.af_generator.save_cache()
            print(f"====> final Atomic facts cache saving took {time() - t:.2f} seconds")

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source,
                                                   cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                score = np.mean([d["is_supported"] for d in decision])

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts))
                    score = penalty * score

                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    t = time()
                    self.save_cache()
                    print(f"====> Intermediate decisions+retrieval cache saving took {time() - t:.2f} seconds")

        t = time()
        self.save_cache()
        print(f"====> final decisions+retrieval cache saving took {time() - t:.2f} seconds")

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)

        return out

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = NonCacheRetrieval(self.db[name], cache_path, embed_cache_path, retrieval_type='bm25', batch_size=self.batch_size)  # <----- this was changed from the orig
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


class MockFactScorer:
    def get_score(self, topics, generations, gamma=None):
        decisions = [[{"atom": generations[0], "is_supported": True}]]
        return {"score": 0.11,
               "respond_ratio": 0.8,
               "decisions": decisions,
               "num_facts_per_response": 0.3}

if os.getenv('OPENAI_KEY') is not None:
    with open('/tmp/oai_key.txt', 'w') as f:
        f.write(os.getenv('OPENAI_KEY'))

    data_dir = cache_dir = model_dir = os.getenv('FACTSCORE_DIR', ".cache/factscore")
    assert os.path.exists(os.path.join(data_dir, 'enwiki-20230401.db')), "You are missing enwiki-20230401.db so either you need to set FACTSCORE_DIR or download the data"

    if os.getenv('MOCK_SCORER', 'False') == 'True':
        fs = MockFactScorer()
    else:
        t = time()
        fs = CustomFactScorer(openai_key='/tmp/oai_key.txt', data_dir=data_dir, cache_dir=cache_dir,  model_dir=model_dir)
        print(f"---> FactScorer loaded in {time() - t:.2f} seconds")
else:
    print('Warning - factscore will not be available unless you define OPENAI_KEY in your env variables')
    fs = None


# same as in run_hallucinations_check.py
BAD_FORMAT = 4
TOPIC_CHANGE = 5
EOS = 6

MIN_PARAGRAPH_LENGTH = 15
def detect_open_ended_topic_change(paragraph, topic, return_before_after=False):
    # It is likely a topic change if we see "The following is a " again.
    # Also, if \n\n then it is possible unless:
    #     The new line is very short
    #      It starts with he,she, it, they, When
    #      It starts with one of the unigrmas in the topic
    #      It starts with the phrase "references" or "Category - in which case we can consider it a valid topic change

    # Here, we may want to check for numbers repetitions (i.e the entire sentence is repeated except for numbers)
    sentences = split_to_sentences(paragraph)
    inds = [i for i, s in enumerate(sentences) if s.lower().startswith('the following is a bio')]
    if len(inds) > 0:
        if return_before_after:
            sent_of_topic_change = sentences[inds[0]]
            before = paragraph[:paragraph.index(sent_of_topic_change)]
            after = paragraph[paragraph.index(sent_of_topic_change):]
            return sentences[:inds[0]], TOPIC_CHANGE, before, after
        return sentences[:inds[0]], TOPIC_CHANGE

    paragraphs = paragraph.split('\n\n')
    for i, p in enumerate(paragraphs[1:], 1):
        if any(p.lower().startswith(w) for w in ['references', 'category']):
            # this is a topic change
            if return_before_after:
                sent_of_topic_change = paragraphs[i]
                before = paragraph[:paragraph.index(sent_of_topic_change)]
                after = paragraph[paragraph.index(sent_of_topic_change):]
                return split_to_sentences('\n\n'.join(paragraphs[:i])), TOPIC_CHANGE, before, after

            return split_to_sentences('\n\n'.join(paragraphs[:i])), TOPIC_CHANGE
        if any(p.lower().startswith(w) for w in ['he', 'she', 'it', 'they', 'when', 'in'] + topic.lower().split()):
            # this is not a topic change, since it continues to discuss our topic (most likely)
            continue
        if len(p) <= MIN_PARAGRAPH_LENGTH:
            # this is not always the case, but very often it is something like a passage
            # title (e.g. References, Discography etc.), which we do consider a topic change of a kind
            if return_before_after:
                sent_of_topic_change = paragraphs[i]
                before = paragraph[:paragraph.index(sent_of_topic_change)]
                after = paragraph[paragraph.index(sent_of_topic_change):]
                return split_to_sentences('\n\n'.join(paragraphs[:i])), TOPIC_CHANGE, before, after

            return split_to_sentences('\n\n'.join(paragraphs[:i])), TOPIC_CHANGE
            # TODO - not sure if it's better to assume a topic change or not

    if not return_before_after:
        if not sentences[-1].strip().endswith('.'):
            return sentences[:-1], BAD_FORMAT  # ended with a bad format because the token count was not enough, don't consider the last sentence

        return sentences, EOS
    return None, None, None, None  # not implemented but who cares

NUMBER_PATTERN = re.compile(r'\d+')
def evaluate_open_ended_generation(paragraph: str, topic: str):
    sentences: List[str]
    sentences, ending = detect_open_ended_topic_change(paragraph, topic)
    
    res = [{"sentence": sentence, "repeating_index": None, "fact_score_result": None} for sentence in sentences]

    # find repetitions
    seen_sentences = {}
    for i, sentence_res in tqdm(enumerate(res)):

        # find if this is a repetition
        sentence = sentence_res["sentence"]
        formatted_sentence = sentence.strip()
        sentence_res["repeating_index"] = seen_sentences.get(formatted_sentence)
        # also, consider a "cleaned version" of the sentence, with all the numbers replaced with a placeholder
        cleaned_sentence = ' '.join([NUMBER_PATTERN.sub('$$$', token) for token in sentence.lower().split()])
        if seen_sentences.get(formatted_sentence) is None:
            sentence_res["repeating_index"] = seen_sentences.get(cleaned_sentence)

        # if it's not a repetition
        if sentence_res["repeating_index"] is None:
            # this is the first time we saw this sentence, so let's cache it
            seen_sentences[sentence] = i
            seen_sentences[cleaned_sentence] = i

            # send to factscore todo we can batch if we want to save speed
            sentence_res["fact_score_result"] = fs.get_score([topic], [sentence], gamma=None)

    # we wish to remove sentences without atoms, but those sentences can be repeated as well, so we would also have to remove the repeated cases, otherwise they point to somewhere non sensical.
    # for repeating index, fact_score_result is None
    final_results = []
    repeating_index_map = np.arange(len(res))
    for i, r in enumerate(res):
        if r['repeating_index'] is not None:
            r['repeating_index'] = repeating_index_map[r['repeating_index']]
            final_results.append(r)
        elif r["fact_score_result"]['decisions'] == [None]:
            warnings.warn(f'Warning - in sentence number {i} ({r["sentence"]}) for topic {topic} there were no atomic facts found. This is likely a mistake., removing the sentence but you should take a look')
            repeating_index_map[i:] -= 1
        else:
            final_results.append(r)

    return final_results, ending
