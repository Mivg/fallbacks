# Datasets

In the paper, we use the following datasets:
- **TriviaFacts**: The dataset is available [here](trivia_facts.jsonl), where each line is a json from a prompt to the list of answers.
- **QAMPARI**: Based on the [Amouyal et al., 2022](https://arxiv.org/abs/2205.12665), 100 multi-answer questions were sampled and rephrased. The dataset is available [here](qampari_rephrased.jsonl), where each line is a json from a prompt to the list of answers.
- **FakeQAMPARI**: Based on the above sample from the QAMPARI dataset, here the subject entity in each question was replaced with a fictitious one. The dataset is available [here](fake_qampari.jsonl), where each line is a json from a prompt to an empty list as there are no correct answers.
- **BIOGENERATION**: Based on the work by [Min et al., 2023](https://arxiv.org/abs/2305.14251), we subsample 25 topics from each popularity level and provide the topics and the respective metadata in [topics.jsonl](topics.jsonl).
