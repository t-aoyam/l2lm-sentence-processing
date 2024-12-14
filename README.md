# L2LM Sentence Processing
This repository contains source code used to train "L2LMs" presented in our EMNLP paper [Modeling Nonnative Sentence Processing with L2 Language Models](https://aclanthology.org/2024.emnlp-main.283/).

## Setting up the Environment
```
$ conda create -n l2lm python=3.8
$ conda activate l2lm
$ pip install -r requirements.txt
```

## Training Second Language Language Models (L2LMs)

### Training on the First Language
```
$ python python train_l2lm.py --l1_fp japanese-cc100 --config_fp data/configs/config-mp.json
```
This will train a mini GPT-2 model on the Japanese subcorpus of CC100, which should take up to 10 hours.

### Training on the Second Langauge

```
$ python train_l2lm.py --l2_fp english-simplewiki --model_train data/models/l2lm-japanese-cc100-mp/l1 --config_fp data/configs/config-mp.json
```
This will train the L1 trained model on L2 (English simple wikipedia corpus).

## Notes

## Citation
When using our work, please use the following citation:
```
@inproceedings{aoyama-schneider-2024-modeling,
    title = "Modeling Nonnative Sentence Processing with {L}2 Language Models",
    author = "Aoyama, Tatsuya  and
      Schneider, Nathan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.283",
    doi = "10.18653/v1/2024.emnlp-main.283",
    pages = "4927--4940",
    abstract = "We study LMs pretrained sequentially on two languages ({``}L2LMs{''}) for modeling nonnative sentence processing. In particular, we pretrain GPT2 on 6 different first languages (L1s), followed by English as the second language (L2). We examine the effect of the choice of pretraining L1 on the model{'}s ability to predict human reading times, evaluating on English readers from a range of L1 backgrounds. Experimental results show that, while all of the LMs{'} word surprisals improve prediction of L2 reading times, especially for human L1s distant from English, there is no reliable effect of the choice of L2LM{'}s L1. We also evaluate the learning trajectory of a monolingual English LM: for predicting L2 as opposed to L1 reading, it peaks much earlier and immediately falls off, possibly mirroring the difference in proficiency between the native and nonnative populations. Lastly, we provide examples of L2LMs{'} surprisals, which could potentially generate hypotheses about human L2 reading.",
}
```
