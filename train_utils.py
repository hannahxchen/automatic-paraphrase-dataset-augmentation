import os
import logging
from random import shuffle

import torch
from torch.utils.data import TensorDataset

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class StratifiedSampler(object):
    def __init__(self, splits, processor, tokenizer, data_dir, max_seq_length, mode='normal'):
        self.mode = mode
        self.splits = splits
        self.n_splits = len(splits)
        self.current_split = None
        self.processor = processor
        
        features = self.load_features(tokenizer, data_dir, max_seq_length, mode='normal')
        
        if self.mode == 'augmented':
            reverse_features = self.load_features(tokenizer, data_dir, max_seq_length, mode='reverse')
            all_features = list(zip(features, reverse_features))
            shuffle(all_features)
            features, reverse_features = zip(*all_features)
            
            self.paraphrase_features_reverse = [f for f in reverse_features if f.label == 1]
            self.nonparaphrase_features_reverse = [f for f in reverse_features if f.label == 0]
        else:
            shuffle(features)
        
        self.paraphrase_features = [f for f in features if f.label == 1]
        self.nonparaphrase_features = [f for f in features if f.label == 0]
        
        splits.insert(0, 0)
        ratio = len(self.paraphrase_features) / len(features)
        increments = [splits[i] - splits[i-1] for i in range(1, self.n_splits-1)]
        
        self.paraphrase_splits = [int(n*ratio) for n in splits[1:]]
        self.nonparaphrase_splits = [i-n for i, n in zip(splits[1:], self.paraphrase_splits)]
    
    def load_features(self, tokenizer, data_dir, max_seq_length, mode='normal'):
        if mode == 'normal':
            cached_feature_file = 'QQP-2/cached_train_128_qqp'
        elif mode == 'reverse':
            cached_feature_file = 'QQP-2/cached_train_128_qqp_reverse'

        if os.path.exists(cached_feature_file):
            logger.info("Loading features from cached file %s", cached_feature_file)
            features = torch.load(cached_feature_file)
        else:
            logger.info("Creating features from dataset file at %s", data_dir)

            examples = self.processor.get_train_examples(data_dir, mode=mode)

            features = convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=self.processor.get_labels(),
                                                    max_length=max_seq_length,
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=0,
            )
            logger.info("Saving features into cached file %s", cached_feature_file)
            torch.save(features, cached_feature_file)
            
        return features
    
    def subsample(self):
        for i in range(self.n_splits):
            self.current_split = self.splits[i+1]
            logger.info("Current split: {}".format(self.current_split))
                         
            features = self.paraphrase_features[:self.paraphrase_splits[i]] + \
            self.nonparaphrase_features[:self.nonparaphrase_splits[i]]
            
            if self.mode == 'augmented':
                reverse_features = self.paraphrase_features_reverse[:self.paraphrase_splits[i]] + \
                self.nonparaphrase_features_reverse[:self.nonparaphrase_splits[i]]
            
                features.extend(reverse_features)
            
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            
            yield dataset


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, output_dir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.output_dir = output_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, args, val_loss, model, step):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(args, val_loss, model, step) 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.save_checkpoint(args, val_loss, model, step) 
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            
    def save_checkpoint(self, args, val_loss, model, step):
        '''Saves model when validation loss decrease.'''
        
        output_dir = os.path.join(self.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                pass
                
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(dict(args), os.path.join(output_dir, 'training_args.bin'))
        self.val_loss_min = val_loss
        
        
def compute_metrics(preds, labels, metric='acc_and_f1', prefix='eval', binary=True):
    assert len(preds) == len(labels)

    if metric == 'acc':
        return (preds == labels).mean()
    elif metric == 'acc_and_f1':
        acc = (preds == labels).mean()
        if binary:
            f1 = f1_score(y_true=labels, y_pred=preds)
        else:
            f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        return {
            "{}_acc".format(prefix): acc,
            "{}_f1".format(prefix): f1,
            "{}_acc_and_f1".format(prefix): (acc + f1) / 2,
        }
    else:
        raise KeyError(metric)