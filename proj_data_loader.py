
import random
import numpy as np
import csv
import os
import sklearn.model_selection as skl

# todo count how many TCRs and peps there are in each set (for TN FN TP FP tables)


def read_data(csv_file):
    with open(csv_file, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)

        tcrs = set()
        peps = set()
        all_pairs = []
        for line in reader:
            tcr, pep,label = line[1], line[0], float(line[2])

            # Proper tcr and peptides
            if any(att == 'NA' or att == "" for att in [tcr, pep]):
                continue
            if any(key in tcr + pep for key in ['#', '*', 'b', 'f', 'y', '~',
                                                'O', '/', '1', 'X', '_', 'B', '7']):
                continue
            all_pairs.append((tcr, pep, label))
    return all_pairs

# PSB: For Project
def load_data(pairs_file_train, pairs_file_test):
    train = read_data(pairs_file_train)
    test = read_data(pairs_file_test)

    random.shuffle(test)
    return train, test


