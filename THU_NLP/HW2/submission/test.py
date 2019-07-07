#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:05:05 2019

@author: Naifu 2018280351
"""

from keras.models import load_model
from utils import *
import os

BASE_DIR = os.getcwd()
TRAIN_PATH = os.path.join(BASE_DIR, 'data/train.txt')
VAL_PATH = os.path.join(BASE_DIR, 'data/dev.txt')
TEST_PATH = os.path.join(BASE_DIR, 'data/test.txt')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

# get train, val, test data
paths = [TRAIN_PATH, VAL_PATH, TEST_PATH]
_, _, _, _, _, test_x, test_y = read_data(paths, max_len=config['MAX_SEQUENCE_LENGTH'])


def eval_models(folder_path, test_x, test_y):
    # evaluates all the weights saved for a model in a model folder, returns the best test acc
    loss_list = []
    acc_list = []
    model_paths = os.listdir(folder_path)
    
    for model_path in model_paths:
        if model_path[0] == '.':
            continue
        else:
            model_path = os.path.join(folder_path, model_path)
            model = load_model(model_path)
            
            # evaluate the model
            loss, acc = model.evaluate(test_x, test_y, verbose=1)
            loss_list.append(loss)
            acc_list.append(acc)
       
    test_loss = min(loss_list)
    test_acc = max(acc_list)
    
    print('{0}: acc {1}%'.format(model.name, round(test_acc*100,2)))
#    print(test_loss)
    
    return test_acc, round(test_loss,4)


folder_paths = os.listdir(WEIGHTS_DIR)
for folder_path in folder_paths:
    if folder_path[0] == '.' or folder_path == 'logs':
        continue
    else:
        folder_path = os.path.join(WEIGHTS_DIR, folder_path)
        eval_models(folder_path, test_x, test_y)