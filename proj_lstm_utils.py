import copy

import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from proj_ERGO_models import DoubleLSTMClassifier, ModifiedDoubleLSTMClassifier
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
# import cuml
# from cuml.svm import SVC

"""
def get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label, weight = pair
        tcrs.append(tcr)
        peps.append(pep)
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs
"""
import numpy as np

def extract_features(model_path,params, batches, device):
    """
    Extracts features using the LSTM layers of the trained model.
    
    Args:
        model (nn.Module): The trained model with LSTM layers.
        batches (list): List of data batches.
        device (torch.device): The device (CPU/GPU) for computation.
        
    Returns:
        np.ndarray: Feature matrix.
        np.ndarray: Corresponding labels.
    """
    model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    if(device == "cuda"):
        model.load_state_dict(torch.load(model_path))
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])

# Load the state dictionary into the model


    # Move the model to the appropriate device (e.g., GPU or CPU)
    model.to(device)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in batches:
            padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
            padded_tcrs, tcr_lens = padded_tcrs.to(device), tcr_lens.to(device)
            padded_peps, pep_lens = padded_peps.to(device), pep_lens.to(device)

            # Extract features
            feature_batch = model(padded_tcrs, tcr_lens, padded_peps, pep_lens, extract_features=True)
            features.append(feature_batch.cpu().numpy())
            labels.extend(batch_signs)
    
    return np.vstack(features), np.array(labels)

def train_svm(features, labels,device, kernel='rbf', C=1.0, gamma='scale'):
    """
    Trains an SVM classifier on the provided features and labels.
    
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Labels corresponding to the features.
        kernel (str): Kernel type for SVM.
        C (float): Regularization parameter.
        gamma (str): Kernel coefficient.
    
    Returns:
        SVC: Trained SVM model.
        StandardScaler: Scaler used for normalization.
    """
    # Normalize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Initialize and train SVM
    if(device == "cuda"):
        from cuml.svm import SVC as cuSVC
        print("Using GPU-accelerated SVM (cuML).")
        svm = cuSVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    else:
        svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    svm.fit(features, labels)
    print("SVM training completed.")
    return svm, scaler

def evaluate_svm(svm, scaler, features, labels):
    """
    Evaluates the trained SVM on test features and labels.
    
    Args:
        svm (SVC): Trained SVM model.
        scaler (StandardScaler): Scaler used for feature normalization.
        features (np.ndarray): Test feature matrix.
        labels (np.ndarray): Test labels.
    
    Returns:
        float: Accuracy of the SVM.
        float: ROC AUC score of the SVM.
    """
    # Normalize features
    features = scaler.transform(features)

    # Predict and evaluate
    predictions = svm.predict(features)
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, svm.predict_proba(features)[:, 1])
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    return accuracy, roc_auc

def Svm_Classifier(batches, test_batches, device, model_path, params):
    print("Extracting training data \n")
    x_train, y_train = extract_features(model_path, params, batches, device)
    print("Extracting test data \n")
    X_test, y_test = extract_features(model_path, params, test_batches, device)
    print("training")
    svm, scaler = train_svm(x_train, y_train,  device, kernel='rbf', C=1.0, gamma='scale')
    print("evaluating")
# Evaluate the SVM
    accuracy, roc_auc = evaluate_svm(svm, scaler, X_test, y_test)
    # print(f"LSTM Embedding given by best model ")
    # print(accuracy)
    # print(roc_auc)

    return svm, accuracy, roc_auc

def Random_forrest_Classifier(batches, test_batches, device, model_path, params):
    print("Extracting training data \n")
    x_train, y_train = extract_features(model_path, params, batches, device)
    print("Extracting test data \n")
    X_test, y_test = extract_features(model_path, params, test_batches, device)
    print("training")
    if (device == "cuda"):
        from cuml.ensemble import RandomForestClassifier as cuRF
        print("Using GPU-accelerated Random Forest.")
        rfc = cuRF(n_estimators=100, max_depth=10, random_state=42, handle='device')
    else:
        rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Higher weight for positive class
    
    rfc.fit(x_train, y_train)
    print("Random Forest training completed.")
    y_pred = rfc.predict(X_test)
    y_prob = rfc.predict_proba(X_test)[:, 1]

    # Classification report
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    accuracy = accuracy_score(y_test, y_pred)
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    return rfc, accuracy, roc_auc




def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        if any(letter.islower() for letter in tcrs[i]):
            print(tcrs[i])
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, signs, batch_size):
    #Get batches from the data
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # Return list of all batches
    return batches


def get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix):
    #Get batches from the data, including last with padding
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    if missing < batch_size:
        padding_tcrs = ['A'] * missing
        padding_peps = ['A' * (batch_size - missing)] * missing
        convert_data(padding_tcrs, padding_peps, amino_to_ix)
        batch_tcrs = tcrs[index:] + padding_tcrs
        batch_peps = peps[index:] + padding_peps
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        batch_signs = [0.0] * batch_size
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches
    pass


def pad_batch(seqs):
    #Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(probs, batch_signs)
        # Compute loss
        #almost Balanced dataset: 1-95910 0-96087
        #weights = batch_signs * 0.84 + (1-batch_signs) * 0.14
        #loss_function.weight = weights
        #PSB
        probs = probs.squeeze()
        loss = loss_function(probs, batch_signs)
        # with open(sys.argv[1], 'a+') as loss_file:
        #    loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, val_batches, device, args, params, test_batches, kIdx):
    #Train and evaluate the model
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    if args['siamese'] is True:
        model = SiameseLSTMClassifier(params['emb_dim'], params['lstm_dim'], device)
    elif args['lstm_type'] == 0:
        model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    elif args['lstm_type'] == 1:
        model = ModifiedDoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    else:
        model = ModifiedDoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device, True)
    if args['restore_epoch']:
        checkpoint = torch.load(args['temp_model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    scheduler = StepLR(optimizer, step_size=args['lr_step'], gamma=args['lr_gamma'])

    # Initialize early stopping variables
    best_auc = 0
    best_acc = 0
    best_roc = None
    best_epoch = 0
    best_thresh = 0
    patience = params['patience']
    epochs_without_improvement = 0

    model_best = None

    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        #train_auc = evaluate(model, batches, device)[0]
        #Use 0.5 threshold
        train_auc, [train_acc, train_prec, train_recall, train_f1, train_thresh], train_roc = evaluate(model, batches, device, 0.5)
        #Pick best threshold
        #train_auc, (train_acc, train_prec, train_recall, train_f1, train_thresh), train_roc = evaluate(model, batches, device)
        print (f"train_auc:{train_auc}, train_acc:{train_acc}, train_prec:{train_prec}, train_recall:{train_recall}, train_f1:{train_f1}, train_thresh:{train_thresh}")
        #print('train auc:', train_auc)

        with open(args['train_auc_file']+'_fold_'+str(kIdx), 'a+') as file:
            file.write(str(train_auc) + '\n')

        #val_auc, roc = evaluate(model, val_batches, device)
        val_auc, [val_acc, val_prec, val_recall, val_f1, val_thresh], val_roc = evaluate(model, val_batches, device, train_thresh)
        print (f"val_auc:{val_auc}, val_acc:{val_acc}, val_prec:{val_prec}, val_recall:{val_recall}, val_f1:{val_f1}, val_thresh:{val_thresh}")
        #print('val auc:', val_auc)
        with open(args['val_auc_file']+'_fold_'+str(kIdx), 'a+') as file:
            file.write(str(val_auc) + '\n')

        if val_acc > best_acc:
        #if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_roc = val_roc
            best_thresh = train_thresh
            best_epoch = epoch  # Save the epoch with the best AUC
            epochs_without_improvement = 0  # Reset the counter
            model_best = copy.deepcopy(model)
            # Save best model checkpoint in case of crash(for recovery)
            torch.save({'model_state_dict': model.state_dict()}, args['temp_model_path']+'_best.pt')
            print(f"Best Model saved at epoch {epoch + 1} to {args['temp_model_path']+'_best.pt'} with thres{best_thresh}")

        else:
            epochs_without_improvement += 1

        if (epoch+1) % params['model_save_occur'] == 0:
            # Save model checkpoint after each epoch (for recovery)
            torch.save({'model_state_dict': model.state_dict()}, args['temp_model_path']+'_epoch_'+str(epoch+1))
            print(f"Model saved at epoch {epoch + 1} to {args['temp_model_path']+'_epoch_'+str(epoch+1)}")


        print('one epoch time:', time.time() - epoch_time)
        # Check if we've reached the patience limit
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1} with no improvement in AUC')
            break
        # Step the scheduler at the end of each epoch to adjust the learning rate
        scheduler.step()
    print(f"Best Model was at epoch {best_epoch + 1}")
    """
    # X_train, y_train = extract_features(model, train_batches, device)
    x_train, y_train = extract_features(model_best, batches, device)
    X_test, y_test = extract_features(model_best, test_batches, device)
    svm, scaler = train_svm(x_train, y_train, kernel='rbf', C=1.0, gamma='scale')

# Evaluate the SVM
    # accuracy, roc_auc = evaluate_svm(svm, scaler, X_test, y_test)
    # print(f"LSTM Embedding given by best model ")
    # print(accuracy)
    # print(roc_auc)
    """

    test_auc, [test_acc, test_prec, test_recall, test_f1, test_thresh], test_roc = evaluate(model_best, test_batches, device, best_thresh)
    print (f"kfold:{kIdx} test_auc:{test_auc}, test_acc:{test_acc}, test_prec:{test_prec}, test_recall:{test_recall}, test_f1:{test_f1}, test_thresh:{test_thresh}")
    return model_best, test_auc, [test_acc, test_prec, test_recall, test_f1, test_thresh], test_roc
    #return model_best, best_auc, best_roc


def evaluate(model, batches, device, threshold=-1.0):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(np.array(batch_signs).astype(int))
        # print(probs.cpu().data.numpy())
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)

    # Find the threshold that maximizes Youden's index (TPR - FPR)
    if threshold == -1.0:
        youden_index = tpr - fpr
        best_threshold_index = np.argmax(youden_index)
        threshold = thresholds[best_threshold_index]

    predicted_labels = (np.array(scores) >= threshold).astype(int)

    #print(true, '\n', predicted_labels.flatten())
    # Calculate metrics
    accuracy = accuracy_score(true, predicted_labels)
    precision = precision_score(true, predicted_labels)
    recall = recall_score(true, predicted_labels)
    f1 = f1_score(true, predicted_labels)

    return auc, [accuracy, precision, recall, f1, threshold], [fpr, tpr, thresholds]
    #return auc, (fpr, tpr, thresholds)


"""
def evaluate_full(model, batches, device):
    model.eval()
    true = []
    scores = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
        batch_size = len(tcr_lens)
        assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        true = true[:index]
        scores = scores[:index]
    if int(sum(true)) == len(true) or int(sum(true)) == 0:
        # print(true)
        raise ValueError
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)
"""


def predict(model, batches, device):
    model.eval()
    preds = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        preds.extend([t[0] for t in probs.cpu().data.tolist()])
        batch_size = len(tcr_lens)
        #assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        preds = preds[:index]
    return preds
