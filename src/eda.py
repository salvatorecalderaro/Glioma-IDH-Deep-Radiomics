import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

dpi = 1000
fontsize = 18
plt.rcParams["text.usetex"] = True


def idh_to_label(idh_value):
    if idh_value.lower() == 'wildtype':
        return 0
    else:
        return 1

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
        print(f"IDH value for {pid}: {idh} -> {label}")
        feats = find_features(raw_data, pid)
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
    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    return train_data, test_data   


def plot_data_dist(train_data, test_data):
    classes = {0: "IDH Wildtype", 1: "IDH Mutant"}
    cmap = plt.cm.Set1
    class_colors = {
        "IDH Mutant": cmap(0),
        "IDH Wildtype": cmap(1)
    }
    text_labels_train = [classes[label] for label in train_data]
    text_labels_test = [classes[label] for label in test_data]

    train_dist = Counter(text_labels_train)
    test_dist = Counter(text_labels_test)
    
    print(train_dist)
    print(test_dist)


    plt.figure(figsize=(12, 5))
    plt.suptitle("Data Distribution - IDH Wildtype vs Mutant", fontsize=fontsize)
    plt.subplot(1, 2, 1)

    plt.bar(
        train_dist.keys(),
        train_dist.values(),
        color=[class_colors[k] for k in train_dist.keys()],
        alpha=0.8
    )

    plt.xlabel("Class", fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)
    plt.title("Training", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.subplot(1, 2, 2)

    plt.bar(
        test_dist.keys(),
        test_dist.values(),
        color=[class_colors[k] for k in test_dist.keys()],
        alpha=0.8
    )

    plt.xlabel("Class", fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)
    plt.title("Test", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("../plots/data_dist_IDH.png", dpi=dpi)
    plt.close()
    
def main():
    train_data, test_data = create_train_test_split()
    plot_data_dist(train_data[1], test_data[1])

if __name__ == "__main__":
    main()