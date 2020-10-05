from __future__ import absolute_import, division, print_function

import sys
import argparse
import glob
import logging
import os
import random
import pickle

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import trange, tqdm

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer, 
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from train_utils import compute_metrics, EarlyStopping
from data_utils import processors, convert_examples_to_features

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """

    # Set seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Freeze the weights of pre-trained layers
    if args.freeze_pretrained:
        for param in model.bert.parameters():
            param.requires_grad = False

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    early_stopping = EarlyStopping(output_dir=args.output_dir, patience=args.patience, verbose=True)
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", miniters=10)
        tr_loss, logging_loss = 0.0, 0.0
        output_label_ids = None
        preds = None
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    logger.info("***** Train results *****")
                    logging.info("Learning rate = {}, Training loss = {}".format(
                        scheduler.get_lr()[0], (tr_loss - logging_loss)/args.logging_steps))
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
#                     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
                        
                    # Log metrics
                    eval_loss = evaluate(args, eval_dataset, model, tokenizer, 
                                         prefix='checkpoint-{}'.format(global_step))
                        
#                     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#                     model_to_save.save_pretrained(output_dir)
#                     torch.save(dict(args), os.path.join(output_dir, 'training_args.bin'))
#                     logger.info("Saving model checkpoint to %s", output_dir)
                    
                    early_stopping(args, eval_loss, model, global_step)
        
                    if early_stopping.early_stop:
                        break
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        if early_stopping.early_stop:
            break
        
        results = {}
        tr_loss /= (step + 1)
        results['training_loss'] = tr_loss
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids, metric=args.eval_metric, prefix='train', binary=args.binary)
        results.update(result)
        
        logger.info("********** Epoch {} **********".format(epoch + 1))
        output_eval_file = os.path.join(args.output_dir, "epoch_{}_eval_results.txt".format(epoch + 1))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results *****")
            writer.write("***** Train results *****\n")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        
        evaluate(args, eval_dataset, model, tokenizer, 
            outfile="epoch_{}_eval_results.txt".format(epoch + 1))
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    torch.save(dict(args), os.path.join(args.output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", args.output_dir)

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss/global_step)

    return model


def evaluate(args, eval_dataset, model, tokenizer, prefix="", 
    outfile="eval_results.txt", output_eval=True):

    eval_output_dir = args.output_dir
    results = {}

    if not os.path.exists(eval_output_dir):
        try:
            os.makedirs(eval_output_dir)
        except:
            pass

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    results = {}
    eval_loss = eval_loss / nb_eval_steps
    results['eval_loss'] = eval_loss
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids, metric=args.eval_metric, binary=args.binary)
    results.update(result)

    if not output_eval:
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
    else:
        output_dir = os.path.join(eval_output_dir, prefix)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                pass

        output_eval_file = os.path.join(output_dir, outfile)

        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            writer.write("***** Eval results *****\n")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    return eval_loss


def predict(args, eval_dataset, model, tokenizer):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    _sigmoid = torch.nn.Sigmoid()
    
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Eval!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    probs = None

    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(**inputs)
            _, logits = outputs[:2]

        if probs is None:
            probs = _sigmoid(logits).detach().cpu().numpy()
        else:
            probs = np.append(probs, _sigmoid(logits).detach().cpu().numpy(), axis=0)

    preds = np.argmax(probs, axis=1)

    return preds, probs


def load_and_cache_examples(args, tokenizer, filename=None, load_type='train'):
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
            load_type, str(args.max_seq_length), str(args.task_name)))
        
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        
        processor = processors[args.dataset]()

        if args.dataset == 'dnli':
            if filename.endswith('jsonl'):
                examples = processor.get_examples(filename, load_type, datatype='json')
            else:
                examples = processor.get_examples(filename, load_type, datatype='tsv')
        else:
            examples = processor.get_examples(filename, load_type)

        label_list = processor.get_labels()
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
        )
                
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def load_and_cache_examples_2(args, tokenizer, filename=None, load_type='train', data_type='is_neutral'):
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
            load_type, str(args.max_seq_length), str(args.dataset)))
        
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        
        processor = processors[args.dataset]()

        if args.dataset == 'dnli':
            if filename.endswith('jsonl'):
                examples = processor.get_examples(filename, load_type, datatype='json')
            else:
                examples = processor.get_examples(filename, load_type, datatype='tsv')
        else:
            examples = processor.get_examples(filename, load_type)

        label_list = processor.get_labels()
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
        )
                
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    
    if data_type == 'is_neutral': 
        label_mapping = {0: 0, 1: 0, 2: 1}

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([label_mapping[f.label] for f in features], dtype=torch.long)

    elif data_type == 'get_relation':
        all_input_ids = torch.tensor([f.input_ids for f in features if f.label != 2], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features if f.label != 2], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features if f.label != 2], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features if f.label != 2], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
