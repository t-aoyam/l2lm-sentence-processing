import os
import argparse
from lm_trainer import LMTrainer
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from torch import cuda
import torch
import glob
import json

def get_scores(model, tokenizer, file_path, device):
    def get_perplexity(model, tokenizer, sentence, device):
        inputs = tokenizer(sentence, return_tensors='pt')
        inputs.to(device)
        loss = model(inputs['input_ids'], labels=inputs['input_ids']).loss
        return torch.exp(loss)

    with open(file_path) as f:
        data = list(f)

    acc = 0
    for item in data:
        line = json.loads(item)
        good = line["sentence_good"]
        bad = line["sentence_bad"]
        good_score = get_perplexity(sentence=good, model=model, tokenizer=tokenizer, device=device)
        bad_score = get_perplexity(sentence=bad, model=model, tokenizer=tokenizer, device=device)
        if bad_score >= good_score:
            acc += 1

    acc = acc / len(data)
    return acc

def evaluate_on_blimp(model, model_path, tokenizer, device):
    # path = "tests/wh_vs_that_with_gap_long_distance.jsonl"
    model.to(device)
    paths = glob.glob(os.path.join("data", "tests", "*.jsonl"))
    model_info = model_path.split(os.path.sep)  # .../model-name/l2 or .../model-name/l2/checkpoint-0000
    if 'checkpoint' in model_info[-1]:
        fn = '-'.join(model_info[-3:])+'.tab'
    elif model_info[-1] == 'l1' or 'l2':
        fn = '-'.join(model_info[-2:])+'.tab'
    else:
        raise IOError(f"the specified model {model_eval} is not named properly.")

    eval_save_fp = os.path.join('data', 'blimp_results', fn)
    results = []
    for path in paths:
        acc = get_scores(model, tokenizer, path, device)
        test_type = path.split(os.path.sep)[-1].split('.')[0]
        results.append([test_type, str(acc)])
        print(path + " " + str(acc * 100))
    with open(eval_save_fp, 'w') as f:
        f.write('\n'.join(['\t'.join(result) for result in results]))

def main(l1_fp, l2_fp, model_train, model_eval, if_eval, output_dir, config_fp, model_name, device):

    if if_eval == 2:
        if not model_eval:
            raise OSError("No model for evaluation specified!")
        model = AutoModelForCausalLM.from_pretrained(model_eval)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_eval)
        evaluate_on_blimp(model, model_eval, tokenizer, device)
        return

    with open(config_fp) as f:
        config_dict = json.load(f)

    for nth in config_dict:
        for cat in config_dict[nth]:
            for key in config_dict[nth][cat]:
                val = config_dict[nth][cat][key]
                config_dict[nth][cat][key] = val if type(val) in [list, float] or not val.replace('_', '').isdigit() else int(val)

    if l1_fp:
        print('\n' + '=' * 100 + f'Training GPT-2 on L1 ({l1_fp}) from scratch...')
        l1_output_dir = os.path.join(output_dir, 'l1')
        if not os.path.exists(l1_output_dir):
            os.mkdir(l1_output_dir)
        l1lm_trainer = LMTrainer(output_dir=l1_output_dir,
                                 model_name=model_name,
                                 input_fp=l1_fp,
                                 config_dict=config_dict['l1'],
                                 model_train=None,
                                 from_hub=True,
                                 **config_dict['l1']['lmtrainer']
                                 )
        l1lm_trainer.train_lm()
        model = AutoModelForCausalLM.from_pretrained(l1_output_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(l1_output_dir)
    else:
        if not model_train:
            raise OSError("No model for L2 training specified!")
        print('\n' + '=' * 100 + f'Loading GPT-2 pre-trained on L1 ({model_train})...')
        model = AutoModelForCausalLM.from_pretrained(model_train)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_train)

    if l2_fp:
        print('\n' + '=' * 100 + f'Training GPT-2 on L2 (English)...')
        l2_output_dir = os.path.join(output_dir, 'l2')
        if not os.path.exists(l2_output_dir):
            os.mkdir(l2_output_dir)
        l2lm_trainer = LMTrainer(output_dir=l2_output_dir,
                                 model_name=model_name,
                                 input_fp=l2_fp,
                                 config_dict=config_dict['l2'],
                                 model_train=model_train,
                                 from_hub=True,
                                 **config_dict['l2']['lmtrainer']
                                 )
        l2lm_trainer.train_lm()
        model = AutoModelForCausalLM.from_pretrained(l2_output_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(l2_output_dir)

    # evaluation
    if if_eval == 1:
        evaluate_on_blimp(model, l2_output_dir, tokenizer, device)

def get_model_info_from_path(path):
    path_as_list = path.split(os.path.sep)  # [.../model-name/l2] OR [.../model-name/l2/checkpoint-0000]
    if 'checkpoint' in path_as_list[-1]:
        model_info = path_as_list[-3]
    elif path_as_list[-1] == 'l1' or 'l2':
        model_info = path_as_list[-2]
    else:
        raise IOError(f"the specified model ({path}) is not named properly.")

    model_info = model_info.split('-')
    if len(model_info) == 5:
        _, l1, l1_corpus_type, l2_corpus_type, config_type = model_info  # l2lm-l1-l1corpus-l2corpus-config
    elif len(model_info) == 4:
        _, l1, l1_corpus_type, config_type = model_info  # l2lm-l1-l1corpus-l2corpus-config
        l2_corpus_type = ''

    return l1, l1_corpus_type, l2_corpus_type, config_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to train mini gpt2 model")
    parser.add_argument("--l1_fp", default="", help="fp to the L1 corpus (e.g. data/corpus/japanese-aochildes)")
    parser.add_argument("--l2_fp", default="", help="fp to the L2 corpus (e.g. data/corpus/english-ads)")
    parser.add_argument("--model_train", default="", help="if only L2 corpus is specified, train this model on L2 (skip redundant L1 training)")
    parser.add_argument("--model_eval", default="", help="if only eval and no training, evaluate on this model")
    parser.add_argument("--eval", default=0, help="0 if no eval, 1 if eval, 2 if only eval and no training")
    parser.add_argument("--use_wandb", action="store_true", help="should I use wandb for logging?")
    parser.add_argument("--gpu", default=0, help="which GPU should I use? 0 or 1, choose wisely based on GPU usage!")
    parser.add_argument("--config_fp", default=None, help="fp to the .json config file for tokenizer and lm training")

    args = parser.parse_args()
    l1_fp = args.l1_fp
    l2_fp = args.l2_fp
    model_train = args.model_train
    model_eval = args.model_eval
    if_eval = int(args.eval)
    use_wandb = args.use_wandb
    gpu = str(args.gpu)
    config_fp = args.config_fp

    if config_fp:
        config_type = config_fp.split('.')[0].split('-')[-1]  # config-cp.json (should be overwritten if model_train)

    l1_corpus = l1_fp.split(os.path.sep)[-1].split('.')[0]
    l2_corpus = l2_fp.split(os.path.sep)[-1].split('.')[0]
    if l1_corpus and model_train:
        raise IOError("Please specify either the L1 corpus to train an LM from scratch, or a pretrained LM.")
    elif l1_corpus:
        l1, l1_corpus_type = l1_corpus.split('-')
    elif model_train:
        if config_fp:
            l1, l1_corpus_type, _, _ = get_model_info_from_path(model_train)
        else:
            l1, l1_corpus_type, _, config_type = get_model_info_from_path(model_train)
    elif model_eval:
        l1, l1_corpus_type, l2_corpus_type, config_type = get_model_info_from_path(model_eval)

    l2_corpus_type = l2_corpus.split('-')[-1]
    if if_eval != 2 and not config_fp:
        raise IOError("if you would like to train a model, please provide a config file!")
    if use_wandb:
        import wandb
        wandb.init(project="l2lm-gpt2", entity="t-aoyam")
    else:
        print('not using wandb')
        os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda:0' if cuda.is_available() else 'cpu'  # always 0 after hiding other GPUs
    if l2_corpus_type:
        model_name = f'l2lm-{l1}-{l1_corpus_type}-{l2_corpus_type}-{config_type}'  # l2lm-l1-l1corpus-l2corpus-config
    else:
        model_name = f'l2lm-{l1}-{l1_corpus_type}-{config_type}'  # l2lm-l1-l1corpus-config
    output_dir = os.path.join("data", "models", model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main(l1_fp, l2_fp, model_train, model_eval, if_eval, output_dir, config_fp, model_name, device)