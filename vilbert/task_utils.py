from io import open
import json
import logging
import os
import sys

from torch.utils.data import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapVal
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb


def ForwardModels(args, task_cfg, iterId, task_count, task_dataloader_train, model, losses):
    pass




def LoadDatasets(args, task_cfg, task_ids):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(task_ids):
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
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(task_ids):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)
        batch_size = task_cfg[task]['batch_size'] // args.gradient_accumulation_steps
        num_workers = int(args.num_workers / len(task_ids)) + 1
        print("Loading %s Dataset with batch size %d" %(task_cfg[task]['name'],batch_size))
        task_datasets_train[name] = DatasetMapTrain[name](
                            task=task_cfg[task]['name'],
                            dataroot=task_cfg[task]['dataroot'],
                            annotations_jsonpath=task_cfg[task]['annotations_jsonpath'],
                            split=task_cfg[task]['train_split'],
                            image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                            gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                            tokenizer=tokenizer, 
                            padding_index=0,
                            max_seq_length=task_cfg[task]['max_seq_length'])

        task_datasets_val[name] = DatasetMapVal[name](
                            task=task_cfg[task]['name'],
                            dataroot=task_cfg[task]['dataroot'],
                            annotations_jsonpath=task_cfg[task]['annotations_jsonpath'],
                            split=task_cfg[task]['val_split'],
                            image_features_reader= task_feature_reader1[task_cfg[task]['features_h5path1']], 
                            gt_image_features_reader= task_feature_reader2[task_cfg[task]['features_h5path2']],
                            tokenizer=tokenizer, 
                            padding_index=0,
                            max_seq_length=task_cfg[task]['max_seq_length'])

        task_dataloader_train[name] = DataLoader(
            task_datasets_train[name],
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_dataloader_val[name] = DataLoader(
            task_datasets_val[name],
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        task_batch_size[name] = batch_size
        task_num_iters[name] = len(task_dataloader_train[name])

    return task_batch_size, task_num_iters, task_names, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val


def LoadModels(args, task_cfg, task_ids, task_datasets_train, ModelMap, LossMap, config):

    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        model_type = task_cfg[task]['type']
        if model_type not in task_types:
            task_types.append(model_type)
        losses[name] = LossMap[task_cfg[task]['loss']]
        
        if num_labels < task_datasets_train[name].num_labels:
            num_labels = task_datasets_train[name].num_labels

    if len(task_types) > 1:
        model_type = 'ALL-tasks'
    else:
        model_type = task_types[0]

    if args.from_pretrained:
        models = ModelMap[model_type].from_pretrained(
            args.from_pretrained, config, num_labels=num_labels
        )
    else:
        models = ModelMap[model_type](
            args.bert_model, config, num_labels=num_labels)

    return models, losses

