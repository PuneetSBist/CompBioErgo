
import random
import numpy as np
import csv
import os
import sklearn.model_selection as skl

# todo count how many TCRs and peps there are in each set (for TN FN TP FP tables)


def read_data(csv_file, is_predict=False):
    all_pairs = []
    if csv_file == "":
        return all_pairs

    with open(csv_file, 'r', encoding='unicode_escape') as file:
        #file.readline()
        reader = csv.reader(file)

        tcrs = set()
        peps = set()
        for line in reader:
            tcr, pep,label = line[1], line[0], float(line[2]) if is_predict == False else 0.0

            # Proper tcr and peptides
            if any(att == 'NA' or att == "" for att in [tcr, pep]):
                continue
            if 'B' in tcr + pep:
                continue
                
            if any(key in tcr + pep for key in ['#', '*', 'b', 'f', 'y', '~',
                                                'O', '/', '1', 'X', '_', 'B', '7']):
                continue
                
            all_pairs.append((tcr, pep, label))
    return all_pairs

# PSB: For Project
def load_data_predict(pairs_file_test, shuff=True):
    test = read_data(pairs_file_test, True)
    if shuff and len(test):
        random.shuffle(test)
    return test

# PSB: For Project
def load_data(pairs_file_train, pairs_file_test, shuff=True):
    train = read_data(pairs_file_train)
    test = read_data(pairs_file_test)

    if shuff and len(train):
        random.shuffle(train)
    if shuff and len(test):
        random.shuffle(test)
    return train, test


