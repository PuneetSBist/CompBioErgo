# THIS IS THE MAIN PYTHON FILE TO RUN
import torch
import pickle
import argparse
import proj_ae_utils as ae
import proj_lstm_utils as lstm
import proj_data_loader
import numpy as np
from proj_ERGO_models import AutoencoderLSTMClassifier, DoubleLSTMClassifier
import csv
import os
from datetime import datetime
from collections import Counter
#import matplotlib.pyplot as plt
from proj_utils import enable_cuda
from joblib import dump
from sklearn.model_selection import KFold



def ae_get_lists_from_pairs(pairs, max_len):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        if len(tcr) >= max_len:
            continue
        tcrs.append(tcr)
        peps.append(pep)
        signs.append(label)
    return tcrs, peps, signs


def lstm_get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        tcrs.append(tcr)
        peps.append(pep)
        signs.append(label)
    # for i in range(50):
    #     tcr, pep, label = pairs[i]
    #     tcrs.append(tcr)
    #     peps.append(pep)
    #     signs.append(label)
    #     tcr, pep, label = pairs[len(pairs) - i-1]
    #     tcrs.append(tcr)
    #     peps.append(pep)
    #     signs.append(label)
    return tcrs, peps, signs


def plot_bar(title, counter):
    keys = list(counter.keys())
    counts = list(counter.values())
    # Create a bar plot
    plt.bar(keys, counts)
    # Adding titles and labels
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Count (Frequency)')
    # Show the plot
    plt.show()


def getOutlier(tcr_counter, pep_counter, title):
    # Extract keys and frequencies
    values1 = list(tcr_counter.elements())  # Expands counter into a list of all values based on their frequency
    values2 = list(pep_counter.elements())  # Expands counter into a list of all values based on their frequency

    mean = np.mean(values1)
    std_dev = np.std(values1)
    lower_threshold1 = mean - 2.0 * std_dev
    upper_threshold1 = mean + 2.0 * std_dev
    print(f"{title}: 2 Sigma Thres({lower_threshold1}, {upper_threshold1})")
    """
    lower_threshold1 = mean - 3.0 * std_dev
    upper_threshold1 = mean + 3.0 * std_dev
    print(f"{title}: 3 Sigma Thres({lower_threshold1}, {upper_threshold1})")
    """

    sorted_values = sorted(values2)
    Q1 = np.percentile(sorted_values, 25)
    Q3 = np.percentile(sorted_values, 75)
    IQR = Q3 - Q1

    lower_threshold2 = Q1 - 1.5*IQR
    upper_threshold2 = Q3 + 1.5*IQR
    print(f"{title}: 1.5 IQR Thres({lower_threshold2}, {upper_threshold2})")
    """
    lower_threshold2 = Q1 - 3.0*IQR
    upper_threshold2 = Q3 + 3.0*IQR
    print(f"{title}: 3 IQR Thres({lower_threshold2}, {upper_threshold2})")
    """

    return lower_threshold1,upper_threshold1,lower_threshold2,upper_threshold2


def run_analytic(merge_list, title, plot=False):
    name1_lengths = [len(name1) for name1, _, _ in merge_list]
    name2_lengths = [len(name2) for _, name2, _ in merge_list]
    intval_values = [intval for _, _, intval in merge_list]

    name1_length_count = Counter(name1_lengths)
    name2_length_count = Counter(name2_lengths)
    intval_count = Counter(intval_values)

    lower_th1, upper_th1, lower_th2, upper_th2 = -1, 1e100, -1, 1e100
    if plot:
        lower_th1, upper_th1, lower_th2, upper_th2 = getOutlier(name1_length_count, name2_length_count, title+'_TCR')

    intval_count = dict(sorted(intval_count.items()))
    name1_length_count = dict(sorted(name1_length_count.items()))
    name2_length_count = dict(sorted(name2_length_count.items()))

    if 0 and plot:
        plot_bar(title+'_TCR', name1_length_count)
        plot_bar(title+'_Pep', name2_length_count)

    print(f"Variation of TCR lengths: {dict(name1_length_count)}")
    print(f"Variation of Pep lengths: {dict(name2_length_count)}")
    print(f"Variation of Label values: {dict(intval_count)}")
    return lower_th1,upper_th1,lower_th2,upper_th2


def run_analytics(train, test):
    merge_list = train + test
    print('*********************Train*********************************')
    lower_th1,upper_th1,lower_th2,upper_th2 = run_analytic(train, 'Train', True)
    print('**********************Test********************************')
    run_analytic(test, 'Test')
    print('**********************Merged****************************')
    run_analytic(merge_list, 'Merge')
    print('******************************************************')

    outlier = []
    for i, value in enumerate(train):
        if len(value[0]) < lower_th1 or len(value[0]) > upper_th1:
            if len(value[1]) < lower_th2 or len(value[1]) > upper_th2:
                outlier.append(value)
    print("Outlier :", len(outlier), outlier)

def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm' or args.model_type == 'svm' or args.model_type == 'rf':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    """
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    """

    # hyper-params
    arg = {}
    arg['train_auc_file'] = args.train_auc_file if args.train_auc_file else 'ignore'
    arg['val_auc_file'] = args.val_auc_file if args.val_auc_file else 'ignore'
    arg['test_auc_file'] = args.test_auc_file if args.test_auc_file else 'ignore'
    arg['temp_model_path'] = args.temp_model_path if args.temp_model_path else 'ignore'
    arg['restore_epoch'] = int(args.restore_epoch) if args.restore_epoch else 0
    arg['lr_step'] = int(args.lr_step)
    arg['kfold'] = int(args.kfold)
    arg['lr_gamma'] = float(args.lr_gamma)

    if args.temp_model_path == 'auto':
        dir = timestamp
        arg['temp_model_path'] = os.path.join(dir, '_'.join([args.model_type, 'psb_checkpoints']))
    if args.val_auc_file == 'auto':
        dir = timestamp
        arg['val_auc_file'] = os.path.join(dir, '_'.join([args.model_type, 'val_auc']))
    if args.train_auc_file == 'auto':
        dir = timestamp
        arg['train_auc_file'] = os.path.join(dir, '_'.join([args.model_type, 'train_auc']))
    if args.test_auc_file == 'auto':
        dir = timestamp
        arg['test_auc_file'] = os.path.join(dir, '_'.join([args.model_type, 'test_auc']))
        #dir = 'save_results'
        #p_key = 'protein' if args.protein else ''
        #arg['test_auc_file'] = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key])
    """
    arg['ae_file'] = args.ae_file
    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
        arg['ae_file'] = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
        pass
    """
    arg['siamese'] = False
    params = {}
    params['lr'] = 1e-4
    params['wd'] = 0    #L2 regularization
    params['epochs'] = 100 #Original
    #params['epochs'] = 200 - arg['restore_epoch']
    params['batch_size'] = 50 #Original
    #params['batch_size'] = 32
    # Number of epochs to wait for improvement
    params['patience'] = 25
    # Save model per epoch in case of crash
    params['model_save_occur'] = 30
    params['lstm_dim'] = 500
    params['emb_dim'] = 10
    #params['dropout'] = 0.2
    params['dropout'] = 0.1 #Original
    params['option'] = 0
    params['enc_dim'] = 100
    params['train_ae'] = True
    print('Params: ', params)

    # Load autoencoder params
    """
    if args.model_type == 'ae':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_' + str(params['enc_dim']) + '.pt'
        arg['ae_file'] = args.ae_file
        checkpoint = torch.load(args.ae_file, map_location=args.device)
        params['max_len'] = checkpoint['max_len']
        params['batch_size'] = checkpoint['batch_size']

    # Load data
    if args.dataset == 'mcpas':
        datafile = r'data/McPAS-TCR.csv'
    elif args.dataset == 'vdjdb':
        datafile = r'data/VDJDB_complete.tsv'
    elif args.dataset == 'united':
        datafile = {'mcpas': r'data/McPAS-TCR.csv', 'vdjdb': r'data/VDJDB_complete.tsv'}
    elif args.dataset == 'tumor':
        datafile = r'tumor/extended_cancer_pairs'
    elif args.dataset == 'nettcr':
        datafile = r'NetTCR/iedb_mira_pos_uniq'
    """
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    train, test = proj_data_loader.load_data(train_data_path, test_data_path)

    #PSB
    #run_analytics(train, test)

    """
    # Save train
    if args.train_data_file == 'auto':
        dir = 'save_results'
        p_key = 'protein' if args.protein else ''
        args.train_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'train'])
    if args.train_data_file:
        with open(args.train_data_file + '.pickle', 'wb') as handle:
            pickle.dump(train, handle)

    # Save test
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test'])
    if args.test_data_file:
        with open(args.test_data_file + '.pickle', 'wb') as handle:
            pickle.dump(test, handle)
    """

    if args.train_auc_file == 'auto' or args.model_file == 'auto' or args.test_auc_file == 'auto' or args.temp_model_path == 'auto':
        os.makedirs(timestamp, exist_ok=True)

    """
    if args.model_type == 'ae':
        # train
        train_tcrs, train_peps, train_signs = ae_get_lists_from_pairs(train, params['max_len'])
        train_batches = ae.get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # test
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, params['max_len'])
        test_batches = ae.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # Train the model
        model, best_auc, best_roc = ae.train_model(train_batches, test_batches, args.device, arg, params)
        pass
    """
    if args.model_type == 'lstm':
        # train
        train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
        lstm.convert_data(train_tcrs, train_peps, amino_to_ix)
        #train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

        # test
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        lstm.convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

        # Train the model
        #model, best_auc, best_roc = lstm.train_model(train_batches, test_batches, args.device, arg, params)
        # Initialize the KFold cross-validator
        kf = KFold(n_splits=arg['kfold'] if arg['kfold'] > 1 else 5, shuffle=True, random_state=42)  # Set shuffle=True for randomness
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_tcrs), 1):
            # Split the data into training and validation sets based on the current fold
            train_tcrs_fold = [train_tcrs[i] for i in train_idx]
            train_peps_fold = [train_peps[i] for i in train_idx]
            train_signs_fold = [train_signs[i] for i in train_idx]

            val_tcrs_fold = [train_tcrs[i] for i in val_idx]
            val_peps_fold = [train_peps[i] for i in val_idx]
            val_signs_fold = [train_signs[i] for i in val_idx]

            # Create batches for training and validation data
            train_batches = lstm.get_batches(train_tcrs_fold, train_peps_fold, train_signs_fold, params['batch_size'])
            val_batches = lstm.get_batches(val_tcrs_fold, val_peps_fold, val_signs_fold, params['batch_size'])

            # Train the model on this fold's training data and evaluate on validation data
            model, best_auc, (test_acc, test_prec, test_recall, test_f1, test_thresh), best_roc = lstm.train_model(train_batches, val_batches, args.device, arg, params, test_batches, fold)
            if arg['kfold'] == 1:
                break

            # Save trained model
            if args.model_file == 'auto':
                dir = timestamp
                args.model_file = os.path.join(dir, '_'.join([args.model_type, 'model']))

            if args.model_file:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'params': params,
                    'best_threshold': test_thresh
                }, args.model_file+'_fold_'+str(fold)+'.pt')

            if args.roc_file:
                # Save best ROC curve and AUC
                np.savez(args.roc_file+'_fold_' + str(fold) , fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))

        pass

    if args.model_type == 'svm':
        print("SVM \n")
        # train
        train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
        lstm.convert_data(train_tcrs, train_peps, amino_to_ix)
        train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])
        # test
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        lstm.convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])
        model_path = args.temp_model_path
        # Train the model
        model, best_auc, best_roc = lstm.Svm_Classifier(train_batches, test_batches, args.device, model_path, params)
        pass
    
    if args.model_type == 'rf':
        print("RF \n")
        # train
        train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
        lstm.convert_data(train_tcrs, train_peps, amino_to_ix)
        train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])
        # test
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        lstm.convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])
        model_path = args.temp_model_path
        # Train the model
        model, best_auc, best_roc = lstm.Random_forrest_Classifier(train_batches, test_batches, args.device, model_path, params)
        pass


    # Save trained model
    ##SVM AND RF ARE SAVED IN A DIFFERENT WAY
    if args.model_type == "svm":
        dump(model, "svm_model.joblib")
    elif args.model_type == "rf":
        dump(model, "rf_model.joblib")
    elif args.model_type == "lstm": 
        if args.model_file == 'auto':
            dir = timestamp
            args.model_file = os.path.join(dir, '_'.join([args.model_type, 'model.pt']))

            #p_key = 'protein' if args.protein else ''
            #args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])
        if args.model_file:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'params': params
                        }, args.model_file)
        
        if args.roc_file:
            # Save best ROC curve and AUC
            np.savez(args.roc_file, fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))
        pass
    


def pep_test(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test.pickle'])
    if args.model_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])

    # Read test data
    with open(args.test_data_file, 'rb') as handle:
        test = pickle.load(handle)

    device = args.device
    if args.model_type == 'ae':
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, 28)
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 30, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        model = DoubleLSTMClassifier(10, 30, 0.1, device)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Get frequent peps list
    if args.dataset == 'mcpas':
        datafile = 'McPAS-TCR.csv'
    p = []
    with open(datafile, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)
        for line in reader:
            pep = line[11]
            if pep == 'NA':
                continue
            p.append(pep)
    d = {t: p.count(t) for t in set(p)}
    sorted_d = sorted(d.items(), key=lambda k: k[1], reverse=True)
    peps = [t[0] for t in sorted_d]
    """
    McPAS most frequent peps
    LPRRSGAAGA  Influenza
    GILGFVFTL   Influenza
    GLCTLVAML   Epstein Barr virus (EBV)	
    NLVPMVATV   Cytomegalovirus (CMV)	
    SSYRRPVGI   Influenza
    """
    rocs = []
    for pep in peps[:50]:
        pep_shows = [i for i in range(len(test_peps)) if pep == test_peps[i]]
        test_tcrs_pep = [test_tcrs[i] for i in pep_shows]
        test_peps_pep = [test_peps[i] for i in pep_shows]
        test_signs_pep = [test_signs[i] for i in pep_shows]
        if args.model_type == 'ae':
            test_batches_pep = ae.get_full_batches(test_tcrs_pep, test_peps_pep, test_signs_pep, tcr_atox, pep_atox, 50, 28)
        if args.model_type == 'lstm':
            lstm.convert_data(test_tcrs_pep, test_peps_pep, amino_to_ix)
            test_batches_pep = lstm.get_full_batches(test_tcrs_pep, test_peps_pep, test_signs_pep, 50, amino_to_ix)
        if len(pep_shows):
            try:
                if args.model_type == 'ae':
                    test_auc, roc = ae.evaluate_full(model, test_batches_pep, device)
                if args.model_type == 'lstm':
                    test_auc, roc = lstm.evaluate_full(model, test_batches_pep, device)
                rocs.append((pep, roc))
                print(str(test_auc))
                # print(pep + ', ' + str(test_auc))
            except ValueError:
                print('NA')
                # print(pep + ', ' 'NA')
                pass
    return rocs


def protein_test(args):
    assert args.protein
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test.pickle'])
    if args.model_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])

    # Read test data
    with open(args.test_data_file, 'rb') as handle:
        test = pickle.load(handle)

    device = args.device
    if args.model_type == 'ae':
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, 28)
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 30, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        model = DoubleLSTMClassifier(10, 30, 0.1, device)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Get frequent peps list
    if args.dataset == 'mcpas':
        datafile = 'McPAS-TCR.csv'
    p = []
    protein_peps = {}
    with open(datafile, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)
        for line in reader:
            pep, protein = line[11], line[9]
            if protein == 'NA' or pep == 'NA':
                continue
            p.append(protein)
            try:
                protein_peps[protein].append(pep)
            except KeyError:
                protein_peps[protein] = [pep]

    d = {t: p.count(t) for t in set(p)}
    sorted_d = sorted(d.items(), key=lambda k: k[1], reverse=True)
    proteins = [t[0] for t in sorted_d]
    """
    McPAS most frequent proteins
    NP177   Influenza
    Matrix protein (M1) Influenza
    pp65    Cytomegalovirus (CMV)
    BMLF-1  Epstein Barr virus (EBV)
    PB1 Influenza
    """
    rocs = []
    for protein in proteins[:50]:
        protein_shows = [i for i in range(len(test_peps)) if test_peps[i] in protein_peps[protein]]
        test_tcrs_protein = [test_tcrs[i] for i in protein_shows]
        test_peps_protein = [test_peps[i] for i in protein_shows]
        test_signs_protein = [test_signs[i] for i in protein_shows]
        if args.model_type == 'ae':
            test_batches_protein = ae.get_full_batches(test_tcrs_protein, test_peps_protein, test_signs_protein, tcr_atox, pep_atox, 50,
                                                   28)
        if args.model_type == 'lstm':
            lstm.convert_data(test_tcrs_protein, test_peps_protein, amino_to_ix)
            test_batches_protein = lstm.get_full_batches(test_tcrs_protein, test_peps_protein, test_signs_protein, 50, amino_to_ix)
        if len(protein_shows):
            try:
                if args.model_type == 'ae':
                    test_auc, roc = ae.evaluate_full(model, test_batches_protein, device)
                if args.model_type == 'lstm':
                    test_auc, roc = lstm.evaluate_full(model, test_batches_protein, device)
                rocs.append((pep, roc))
                # print(protein)
                print(str(test_auc))
                # print(protein + ', ' + str(test_auc))
            except ValueError:
                # print(protein)
                print('NA')
                # print(protein + ', ' 'NA')
                pass
    return rocs


def predict(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    """
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # if args.ae_file == 'auto':
    args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_100.pt'
    """
    if args.model_file == 'auto':
        return
        """
        dir = 'models'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])
        """
    if args.test_data_file == 'auto':
        return
        #args.test_data_file = 'pairs_example.csv'

    # Read test data
    """
    tcrs = []
    peps = []
    signs = []
    max_len = 28
    with open(args.test_data_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            tcr, pep = line
            if args.model_type == 'ae' and len(tcr) >= max_len:
                continue
            tcrs.append(tcr)
            peps.append(pep)
            signs.append(0.0)
    tcrs_copy = tcrs.copy()
    peps_copy = peps.copy()
    """
    test_data_path = args.test_data_path
    _, test = proj_data_loader.load_data("", test_data_path, False)
    tcrs, peps, signs = lstm_get_lists_from_pairs(test)
    tcrs_copy = tcrs.copy()
    peps_copy = peps.copy()

    """
    # Load model
    device = args.device
    if args.model_type == 'ae':
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 100, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    """
    if args.model_type == 'lstm':
        checkpoint = torch.load(args.model_file, map_location=device)
        best_threshold = checkpoint['best_threshold']
        params = checkpoint['params']
        model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
        #model = DoubleLSTMClassifier(10, 500, 0.1, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Predict
    batch_size = params['batch_size']
    """
    if args.model_type == 'ae':
        test_batches = ae.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_len)
        preds = ae.predict(model, test_batches, device)
    """
    if args.model_type == 'lstm':
        lstm.convert_data(tcrs, peps, amino_to_ix)
        #test_batches = lstm.get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix)
        test_batches = lstm.get_batches(tcrs, peps, signs, params['batch_size'])
        #preds = lstm.predict(model, test_batches, device, best_threshold)
    #TODO: change so that without label we also provide the probality
    test_auc, (test_acc, test_prec, test_recall, test_f1, test_thresh), test_roc = evaluate(model, test_batches, device, best_threshold)
    print (f"Predict:{args.test_data_path} using mode:{args.model_file} test_auc:{test_auc}, test_acc:{test_acc}, test_prec:{test_prec}, test_recall:{test_recall}, test_f1:{test_f1}, test_thresh:{test_thresh}")
    test_auc, (test_acc, test_prec, test_recall, test_f1, test_thresh), test_roc = evaluate(model, test_batches, device, 0.5)
    print (f"Predict:{args.test_data_path} using mode:{args.model_file} test_auc:{test_auc}, test_acc:{test_acc}, test_prec:{test_prec}, test_recall:{test_recall}, test_f1:{test_f1}, test_thresh:0.5")
    # Print predictions
    #for tcr, pep, pred in zip(tcrs_copy, peps_copy, preds):
    #    print('\t'.join([tcr, pep, str(pred)]))


if __name__ == '__main__':
    debug = False

    parser = argparse.ArgumentParser()
    if not debug:
        #train or predict
        parser.add_argument("function")
        # ALways lstm
        #parser.add_argument("model_type")
        #cpu or cuda or gpu etc
        parser.add_argument("device")
    parser.add_argument("--train_data_path")
    parser.add_argument("--test_data_path")
    #Save Model for train or load for predict
    parser.add_argument("--kfold", default=1)
    parser.add_argument("--model_file", default='auto' )
    parser.add_argument("--train_auc_file", default='auto' )
    parser.add_argument("--val_auc_file", default='auto' )
    parser.add_argument("--test_auc_file", default='auto' )
    parser.add_argument("--temp_model_path", default='auto' )
    parser.add_argument("--restore_epoch", default=0)
    parser.add_argument("--lr_step", default=5)
    parser.add_argument("--lr_gamma", default=1.0)
    parser.add_argument("--roc_file")
    """
    parser.add_argument("sampling")
    parser.add_argument("--protein", action="store_true")
    parser.add_argument("--hla", action="store_true")
    parser.add_argument("--ae_file")
    parser.add_argument("--train_data_file")
    parser.add_argument("--test_data_file")
    """
    args = parser.parse_args()

    args.model_type = 'rf'
    args.temp_model_path = 'D:\ASU 1-1\Algo in Comp Bio\CompBioErgo-main\RunData\TCR_split_32Batch_Drop20\lstm_model.pt'

    if debug:
        args.function = 'train'
        #args.device = 'cpu'
        args.device = 'cuda'
        """
        args.train_data_path = 'proj_data\\BAP\\tcr_split\\train.csv'
        args.test_data_path = 'proj_data\\BAP\\tcr_split\\test.csv'
        """
        """
        args.train_data_path = '/home/pbist/AlgoCompBio/data/BAP/tcr_split/train.csv'
        args.test_data_path = '/home/pbist/AlgoCompBio/data/BAP/tcr_split/test.csv'
        print("Using TCR split")
        """
        args.train_data_path = '/home/pbist/AlgoCompBio/data/BAP/epi_split/train.csv'
        args.test_data_path = '/home/pbist/AlgoCompBio/data/BAP/epi_split/test.csv'
        args.roc_file = "roc_file"

    print('Default Model')
    print('Default Embedding')
    print('Default LossFunc')
    print('Original Without EarlySTopping')
    print(f'Step LR params: step{args.lr_step}, gamma:{args.lr_gamma}')
    print(f'kfold: {args.kfold}')
    print(f'Using Training set {args.train_data_path} and test {args.test_data_path}')
    if args.device == 'cuda':
        enable_cuda(no_cuda=False)
    if args.function == 'train':
        main(args)
    elif args.function == 'predict':
        predict(args)
    """
    elif args.function == 'test' and not args.protein:
        pep_test(args)
    elif args.function == 'test' and args.protein:
        protein_test(args)
    """

# example
#  python ERGO.py train lstm mcpas specific cuda:0 --model_file=model.pt --train_data_file=train_data --test_data_file=test_data
