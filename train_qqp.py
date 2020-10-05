import os
import random
import pickle
import argparse

import torch
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from data_utils import processors
from run_classifier import train, evaluate, load_and_cache_examples


if not os.path.exists('models'):
    os.makedirs('models')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True, type=str)
args = parser.parse_args()

if args.task_name == 'original':
	task_name = 'qqp'
    train_file = 'data/train.tsv'
    dev_file = 'data/dev.tsv'
	logging_step = 1000

elif args.task_name == 'original_flipped':
	task_name = 'qqp_flipped'
    train_file = 'data/train_orig_flipped.tsv'
    dev_file = 'data/dev_orig_flipped.tsv'
	logging_step = 1000

elif args.task_name == 'augmented':
	task_name = 'qqp_augmented'
    train_file = 'data/train_augmented.tsv'
    dev_file = 'data/dev_augmented.tsv'
	logging_step = 2000

elif args.task_name == 'augmented_flipped':
	task_name = 'qqp_augmented_flipped'
    train_file = 'data/train_augmented_flipped.tsv'
    dev_file = 'data/dev_augmented_flipped.tsv'
	logging_step = 2000

index = 1

while os.path.exists("models/bert_{}_{}_{}".format(dataset, task_name, index)):
    index += 1

output_dir = "models/bert_{}_{}_{}".format(dataset, task_name, index)


args = {
    'seed': random.randint(1, 100),
    'model_type': 'bert',
    'task_name': task_name,
    'freeze_pretrained': False,
    'max_seq_length': 128,
    'n_gpu': torch.cuda.device_count(),
    'per_gpu_train_batch_size': 8,
    'per_gpu_eval_batch_size': 8,
    'num_train_epochs': 5,
    'max_steps': -1,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'weight_decay': 0.0,
    'warmup_steps': 0,
    'eval_metric': 'acc_and_f1',
    'logging_steps': logging_step,
    'save_steps': logging_step,
    'patience': 7,
    'data_dir': 'datasets/QQP-2',
    'output_dir': output_dir,
    'overwrite_cache': False
}

args = dotdict(args)



processor = processors['qqp']()
label_list = processor.get_labels()
args.num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=args.num_labels)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
model.to(args.device)

train_dataset = load_and_cache_examples(args, tokenizer, filename=train_file, load_type='train')
eval_dataset = load_and_cache_examples(args, tokenizer, filename=dev_file, load_type='dev')

model = train(args, train_dataset, eval_dataset, model, tokenizer) 

