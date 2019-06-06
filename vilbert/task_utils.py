from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapVal
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

LossMap = {'BCEWithLogitLoss': nn.BCEWithLogitsLoss(reduction='mean'),
            }

def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, question_ids = batch

    vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
    vision_logit, linguisic_prediction, linguisic_logit = model(question, features, spatials, segment_ids, input_mask, image_mask)

    if task_cfg[task_id]['type'] == 'VL-classifier':
        loss = task_losses[task_id](vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return loss.item(), batch_score.item()

def ForwardModelsTrain(args, task_cfg, device, task_id, iterId, task_count, task_iter_train, task_dataloader_train, model, task_losses):
    # given the current task, decided whether to forward the model and forward with specific loss.
    start_iteration = task_cfg[task_id]['start_iteration']
    if iterId >= start_iteration:
        # reset the task iteration when needed.
        task_count[task_id] += 1
        if iterId % task_count[task_id] == 0:
            task_iter_train[task_id] = iter(task_dataloader_train[task_id])
        # get the batch
        batch = task_iter_train[task_id].next()
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_ids = batch
        # reshape if the data has 3 dimensions.

        # get the model output
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, \
        vision_logit, linguisic_prediction, linguisic_logit = model(question, features, spatials, segment_ids, input_mask, image_mask)

        # for different task, we use different output to calculate the loss.
        if task_cfg[task_id]['type'] == 'VL-classifier':
            loss = task_losses[task_id](vil_prediction, target)
            loss = loss.mean() * target.size(1)
            batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return loss, batch_score


def LoadLosses(args, task_cfg, task_ids):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = 'TASK' + task_id
        model_type = task_cfg[task]['type']
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]['loss']]

    return losses


def LoadDatasets(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        if task_cfg[task]['features_h5path1'] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]['features_h5path1']] = None
        if task_cfg[task]['features_h5path2'] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]['features_h5path2']] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != '':
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(features_h5path, args.in_memory)
    
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != '':
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(features_h5path, args.in_memory)
    
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_names = []
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_ids.append(task)
        task_names.append(name)
        batch_size = task_cfg[task]['batch_size'] // args.gradient_accumulation_steps
        num_workers = int(args.num_workers / len(ids)) + 1
        logger.info("Loading %s Dataset with batch size %d" %(task_cfg[task]['name'],batch_size))
        task_datasets_train[task] = DatasetMapTrain[task](
                            task=task_cfg[task]['name'],
                            dataroot=task_cfg[task]['dataroot'],
                            annotations_jsonpath=task_cfg[task]['annotations_jsonpath'],
                            split=task_cfg[task]['train_split'],
                            image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                            gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                            tokenizer=tokenizer, 
                            padding_index=0,
                            max_seq_length=task_cfg[task]['max_seq_length'])

        task_datasets_val[task] = DatasetMapVal[task](
                            task=task_cfg[task]['name'],
                            dataroot=task_cfg[task]['dataroot'],
                            annotations_jsonpath=task_cfg[task]['annotations_jsonpath'],
                            split=task_cfg[task]['val_split'],
                            image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                            gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                            tokenizer=tokenizer, 
                            padding_index=0,
                            max_seq_length=task_cfg[task]['max_seq_length'])

        task_dataloader_train[task] = DataLoader(
            task_datasets_train[task],
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        task_batch_size[task] = batch_size
        task_num_iters[task] = len(task_dataloader_train[task])

    return task_batch_size, task_num_iters, task_names, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

