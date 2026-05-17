from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml
from sklearn.metrics import f1_score,roc_auc_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix
import pickle
import random
import os

SEED = 2026


def set_seed():
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
def normalize_pid(pid):
    if pid.startswith("UCSF-PDGM-"):
        numeric_part = pid.split('-')[-1].split('_')[0]
    else:
        numeric_part = pid.split('_')[0]
    num = int(numeric_part)
    return f"UCSF-PDGM-{num:03d}"

def find_features(data, pid):
    df = data[data['PazienteID'] == pid]
    radiomics_feats = df.loc[:, ~df.columns.str.startswith("diagnostics_")].drop(columns=["label","label_value","PazienteID"])
    radiomics_feats = np.array(radiomics_feats)
    radiomics_feats = radiomics_feats[:, 1:]
    if radiomics_feats.shape[0] !=8:
        missing_info = np.zeros((8 - radiomics_feats.shape[0], radiomics_feats.shape[1]))
        radiomics_feats = np.vstack((radiomics_feats, missing_info))
    radiomics_feats = radiomics_feats.astype(np.float32)
    return radiomics_feats

def idh_to_label(idh_value):
    if idh_value.lower() == 'wildtype':
        return 0
    else:
        return 1
    
def create_train_test_split():
    raw_data = pd.read_csv("../data/UCSF_Features.csv")
    split = pd.read_csv("../data/patient_splits.csv")
    
    x_train, x_test = [], []
    y_train, y_test = [], []
    
    metadata_file = "../data/UCSF-PDGM-metadata_v5.csv"
    metadata = pd.read_csv(metadata_file)
    for index, row in split.iterrows():
        pid = row['patient_id']
        group = row['set']
        correct_pid =  normalize_pid(pid)
        filtered_metadata = metadata[metadata['ID'] == correct_pid]
        idh = filtered_metadata['IDH'].values[0]
        label = idh_to_label(idh)
        feats = find_features(raw_data, pid)
        feats = feats.flatten()
        if group == 'train':
            x_train.append(feats)
            y_train.append(label)
        else:
            x_test.append(feats)
            y_test.append(label)
    x_train = np.stack(x_train, axis=0)
    x_test = np.stack(x_test, axis=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"Number of patients in training set: {len(y_train)}")
    print(f"Number of patients in test set: {len(y_test)}")
    feature_dim = x_train.shape[1]
    print(f"Number of features: {feature_dim}")
    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    return train_data, test_data   


def apply_random_forest(train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf.fit(x_train, y_train)
    
    path = "../models/random_forest.pkl"
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    y_pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1]
    return y_pred, proba
    

def evaluate_model(targets, predictions,proba):
    f1s = f1_score(targets, predictions, average='micro')
    print(f"F1 Score (Micro): {f1s:.4f}")
    
    acc = accuracy_score(targets, predictions)
    print(f"Accuracy: {acc:.4f}")

    balanced_acc = balanced_accuracy_score(targets, predictions)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")


    recall = recall_score(targets, predictions)
    print(f"Sensitivity (Recall): {recall:.4f}")

    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity: {specificity:.4f}")
    
    auroc = roc_auc_score(targets,proba)
    print(f"AUROC: {auroc:.4f}")
    
    report = {
        "F1 Score (Micro)": float(f1s),
        "Accuracy":float(acc),
        "Balanced Accuracy": float(balanced_acc),
        "Sensitivity (Recall)": float(recall),
        "Specificity": float(specificity),
        "AUROC": float(auroc)
    }
    
    path = "../results/results_random_forest.yaml"
    with open(path, "w") as f:
        yaml.dump(report, f)

def main():
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"IDH Prediction using Random Forest")
    train_data, test_data = create_train_test_split()
    y_pred, proba = apply_random_forest(train_data, test_data)
    targets = test_data[1]
    evaluate_model(targets, y_pred,proba)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return
    
    
if __name__ == "__main__":
    main()