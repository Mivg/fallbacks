import json
import random
from collections import defaultdict

if __name__ == '__main__':
    topics_per_freq = 25
    seed = 42
    with open('data/open_ended/all_topics.jsonl') as f:
        lines = [json.loads(line) for line in f.readlines()]
    random.seed(seed)

    freqs = defaultdict(list)
    for line in lines:
        if len(freqs[line['popularity']]) < topics_per_freq:
            freqs[line['popularity']].append(line)
    assert all(len(freqs[freq]) == topics_per_freq for freq in freqs)
    assert len(freqs) == 5

    with open('data/open_ended/topics.jsonl', 'w') as f:
        for freq in freqs:
            for topic in freqs[freq]:
                f.write(json.dumps(topic) + '\n')
