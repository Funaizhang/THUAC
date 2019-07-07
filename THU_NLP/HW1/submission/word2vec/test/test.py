#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:35:35 2019

@author: Naifu
"""

from nltk.metrics import spearman_correlation, ranks_from_scores
from scipy.stats import spearmanr
import csv

def read_sim_list(filename):
    sim_list_human = []
    sim_list1 = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[3] != 0:
                sim_list_human.append(row[2])
                sim_list1.append(row[3])
    return sim_list_human, sim_list1


sim_score_human, sim_score1 = read_sim_list('output_300_6c.csv')

        
#sim_rank_human = ranks_from_scores(sim_score_list_human)
#sim_rank1 = ranks_from_scores(sim_score_list1)
print(spearmanr(sim_score_human, sim_score1))