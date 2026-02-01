from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
import importlib
from collections import defaultdict
from rdkit import Chem
from collections import defaultdict
import torch
import time
import torch.nn.functional as F
import os
import random
import copy

warnings.filterwarnings("ignore")


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(
        X, Y, train_size=2 / 3, random_state=1203
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_eval, y_eval, test_size=0.5, random_state=1203
    )

    return x_train, x_eval, x_test, y_train, y_eval, y_test


def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)
    ]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


# ========================================
# 评估指标计算函数
# ========================================


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)

        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def ddi_rate_score(record, path):
    ddi_A = dill.load(open(path, "rb"))
    all_cnt = 0
    dd_cnt = 0

    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1

    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt




def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]

    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))
            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    MPNNSet = []
    average_index = []

    for index, atc3 in med_voc.items():
        smilesList = list(molecule[atc3])
        counter = 0
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)
                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                continue
        average_index.append(counter)

    N_fingerprint = len(fingerprint_dict)
    n_col = sum(average_index)
    n_row = len(average_index)
    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        if item > 0:
            average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)


# ========================================

def get_group_weight(data, function_name='IFW', hyper_param=1, merge_type="max"):
    disease_count = defaultdict(int)
    total_diseases = 0

    for patient in data:
        for visit in patient:
            diseases = visit[3]
            for disease in diseases:
                disease_count[disease] += 1
                total_diseases += 1

    disease_frequent = {disease: count / total_diseases for disease, count in disease_count.items()}

    for patient in data:
        for visit in patient:
            diseases = visit[3]

            disease_weights = []
            for disease in diseases:
                disease_freq = disease_frequent.get(disease, 0)
                if function_name == 'IFW':
                    disease_weight = IFW(disease_freq)
                elif function_name == 'LogW':
                    disease_weight = LogW(disease_freq)
                elif function_name == 'ExpW':
                    disease_weight = ExpW(disease_freq, hyper_param)
                disease_weights.append(disease_weight)

            if merge_type == "max":
                final_weight = max(disease_weights) if disease_weights else 0
            elif merge_type == "min":
                final_weight = min(disease_weights) if disease_weights else 0
            elif merge_type == "average":
                final_weight = np.mean(disease_weights) if disease_weights else 0
            elif merge_type == "weighted_average":
                weighted_score = [disease_frequent.get(disease, 0) for disease in diseases]
                total_weight = sum(weighted_score)
                if total_weight > 0:
                    final_weight = sum(w * dw / total_weight for w, dw in zip(weighted_score, disease_weights))
                else:
                    final_weight = 0
            elif merge_type == "sum":
                total_weight = sum(disease_weights) if disease_weights else 0
                final_weight = total_weight / len(diseases) if diseases else 0

            visit[5] = final_weight

    return data

def IFW(pi_g, epsilon=1e-8):
    """
    Inverse Frequency Weight (IFW).
    
    Args:
        pi_g (float): The frequency of class g.
        epsilon (float): A small constant to ensure numerical stability (default is 1e-8).
    
    Returns:
        float: The weight for class g.
    """
    return (pi_g + epsilon) ** -1

def LogW(pi_g):
    """
    Logarithmic Weight (LogW).
    
    Args:
        pi_g (float): The frequency of class g.
    
    Returns:
        float: The weight for class g.
    """
    return np.log(1 + (1 / pi_g))

def ExpW(pi_g, beta=1.0):
    """
    Exponentially Decaying Weight (ExpW).
    
    Args:
        pi_g (float): The frequency of class g.
        beta (float): A parameter that controls the decay rate (default is 1.0).
    
    Returns:
        float: The weight for class g.
    """
    return (1 - pi_g) ** beta


# ========================================

def print_head_tail_split(data):
    head_cnt = 0
    tail_cnt = 0

    for input in data:
        for adm in input:
            if adm[4] == "head":
                head_cnt += 1
            elif adm[4] == "tail":
                tail_cnt += 1

    total_cnt = head_cnt + tail_cnt
    print(f"\nTotal visits: {total_cnt}, Head visits: {head_cnt}, Tail visits: {tail_cnt}")


def history_update_and_print(history, overall_metrics, tail_metrics, head_metrics, epoch):
    
    history["ja"].append(overall_metrics[1])
    history["ddi_rate"].append(overall_metrics[0])
    history["avg_p"].append(overall_metrics[3])
    history["avg_r"].append(overall_metrics[4])
    history["avg_f1"].append(overall_metrics[5])
    history["prauc"].append(overall_metrics[2])
    history["med"].append(overall_metrics[6])

    history["ja_tail"].append(tail_metrics[1])
    history["ddi_rate_tail"].append(tail_metrics[0])
    history["avg_p_tail"].append(tail_metrics[3])
    history["avg_r_tail"].append(tail_metrics[4])
    history["avg_f1_tail"].append(tail_metrics[5])
    history["prauc_tail"].append(tail_metrics[2])
    history["med_tail"].append(tail_metrics[6])

    history["ja_head"].append(head_metrics[1])
    history["ddi_rate_head"].append(head_metrics[0])
    history["avg_p_head"].append(head_metrics[3])
    history["avg_r_head"].append(head_metrics[4])
    history["avg_f1_head"].append(head_metrics[5])
    history["prauc_head"].append(head_metrics[2])
    history["med_head"].append(head_metrics[6])

    if epoch >= 5:
        print(
            "Overall - DDI: {:.4}, Med: {:.4}, Ja: {:.4}, F1: {:.4}, PRAUC: {:.4}".format(
                np.mean(history["ddi_rate"][-5:]),
                np.mean(history["med"][-5:]),
                np.mean(history["ja"][-5:]),
                np.mean(history["avg_f1"][-5:]),
                np.mean(history["prauc"][-5:]),
            )
        )
        print(
            "Tail - DDI: {:.4}, Med: {:.4}, Ja: {:.4}, F1: {:.4}, PRAUC: {:.4}".format(
                np.mean(history["ddi_rate_tail"][-5:]),
                np.mean(history["med_tail"][-5:]),
                np.mean(history["ja_tail"][-5:]),
                np.mean(history["avg_f1_tail"][-5:]),
                np.mean(history["prauc_tail"][-5:]),
            )
        )
        print(
            "Head - DDI: {:.4}, Med: {:.4}, Ja: {:.4}, F1: {:.4}, PRAUC: {:.4}".format(
                np.mean(history["ddi_rate_head"][-5:]),
                np.mean(history["med_head"][-5:]),
                np.mean(history["ja_head"][-5:]),
                np.mean(history["avg_f1_head"][-5:]),
                np.mean(history["prauc_head"][-5:]),
            )
        )
    return history

# ========================================

def model_eval(model_name, model, data_eval, voc_size, dataset_path, epoch):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    ja_tail, prauc_tail, avg_p_tail, avg_r_tail, avg_f1_tail = [[] for _ in range(5)]
    ja_head, prauc_head, avg_p_head, avg_r_head, avg_f1_head = [[] for _ in range(5)]

    med_cnt, visit_cnt = 0, 0
    med_cnt_tail, visit_cnt_tail = 0, 0
    med_cnt_head, visit_cnt_head = 0, 0

    smm_record = []
    smm_record_tail = []
    smm_record_head = []

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        y_gt_tail, y_pred_tail, y_pred_prob_tail, y_pred_label_tail = [], [], [], []
        y_gt_head, y_pred_head, y_pred_prob_head, y_pred_label_head = [], [], [], []

        if model_name == "Retain" and len(input) < 2:
            continue

        for adm_idx, adm in enumerate(input):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            if model_name in ["SafeDrug", "GAMENet", "Retain"]:
                if model_name in ["SafeDrug"]:
                    target_output, _ = model(input[: adm_idx + 1])
                elif model_name in ["GAMENet", "Retain"]:
                    target_output = model(input[: adm_idx + 1])
                    
                target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)

                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))

                if adm[4] == "tail":
                    y_gt_tail.append(y_gt_tmp)
                    y_pred_tail.append(y_pred_tmp)
                    y_pred_prob_tail.append(target_output)
                    y_pred_label_tail.append(sorted(y_pred_label_tmp))
                    visit_cnt_tail += 1
                    med_cnt_tail += len(y_pred_label_tmp)

                elif adm[4] == "head":
                    y_gt_head.append(y_gt_tmp)
                    y_pred_head.append(y_pred_tmp)
                    y_pred_prob_head.append(target_output)
                    y_pred_label_head.append(sorted(y_pred_label_tmp))
                    visit_cnt_head += 1
                    med_cnt_head += len(y_pred_label_tmp)
                
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

            elif model_name in ["Leap"]:
                target_output = model(adm)
                target_output = target_output.detach().cpu().numpy()

                out_list, sorted_predict = sequence_output_process(
                    target_output, [voc_size[2], voc_size[2] + 1]
                )
                y_pred_label.append(sorted(sorted_predict))
                y_pred_prob.append(np.mean(target_output[:, :-2], axis=0))

                y_pred_tmp = np.zeros(voc_size[2])
                y_pred_tmp[out_list] = 1
                y_pred.append(y_pred_tmp)

                if adm[4] == "tail":
                    y_gt_tail.append(y_gt_tmp)
                    y_pred_tail.append(y_pred_tmp)
                    y_pred_prob_tail.append(np.mean(target_output[:, :-2], axis=0))
                    y_pred_label_tail.append(sorted(sorted_predict))
                    visit_cnt_tail += 1
                    med_cnt_tail += len(sorted_predict)

                elif adm[4] == "head":
                    y_gt_head.append(y_gt_tmp)
                    y_pred_head.append(y_pred_tmp)
                    y_pred_prob_head.append(np.mean(target_output[:, :-2], axis=0))
                    y_pred_label_head.append(sorted(sorted_predict))
                    visit_cnt_head += 1
                    med_cnt_head += len(sorted_predict)

                visit_cnt += 1
                med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        if model_name in ["SafeDrug", "GAMENet", "Retain"]:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
        elif model_name in ["Leap"]:
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), y_pred_label,
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

        if len(y_gt_tail) > 0:
            if model_name in ["SafeDrug", "GAMENet", "Retain"]:
                tail_ja, tail_prauc, tail_avg_p, tail_avg_r, tail_avg_f1 = multi_label_metric(
                    np.array(y_gt_tail), np.array(y_pred_tail), np.array(y_pred_prob_tail)
                )
            elif model_name in ["Leap"]:
                tail_ja, tail_prauc, tail_avg_p, tail_avg_r, tail_avg_f1 = sequence_metric(
                    np.array(y_gt_tail), np.array(y_pred_tail), np.array(y_pred_prob_tail), y_pred_label_tail,
                )
            ja_tail.append(tail_ja)
            prauc_tail.append(tail_prauc)
            avg_p_tail.append(tail_avg_p)
            avg_r_tail.append(tail_avg_r)
            avg_f1_tail.append(tail_avg_f1)

            smm_record_tail.append(y_pred_label_tail)

        if len(y_gt_head) > 0:
            if model_name in ["SafeDrug", "GAMENet", "Retain"]:
                head_ja, head_prauc, head_avg_p, head_avg_r, head_avg_f1 = multi_label_metric(
                    np.array(y_gt_head), np.array(y_pred_head), np.array(y_pred_prob_head)
                )
            elif model_name in ["Leap"]:
                head_ja, head_prauc, head_avg_p, head_avg_r, head_avg_f1 = sequence_metric(
                    np.array(y_gt_head), np.array(y_pred_head), np.array(y_pred_prob_head), y_pred_label_head,
                )
            ja_head.append(head_ja)
            prauc_head.append(head_prauc)
            avg_p_head.append(head_avg_p)
            avg_r_head.append(head_avg_r)
            avg_f1_head.append(head_avg_f1)

            smm_record_head.append(y_pred_label_head)

        # llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    ddi_path = os.path.join(dataset_path, "ddi_A_final.pkl")
    ddi_rate = ddi_rate_score(smm_record, path=ddi_path)
    ddi_rate_tail = ddi_rate_score(smm_record_tail, path=ddi_path)
    ddi_rate_head = ddi_rate_score(smm_record_head, path=ddi_path)

    print_head_tail_split(data_eval)

    llprint(
            "\nOverall Performance:\n DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt,
            )
        )
    if visit_cnt_tail > 0:
        llprint(
            "\nTail Performance:\n DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate_tail, np.mean(ja_tail), np.mean(prauc_tail), np.mean(avg_p_tail), np.mean(avg_r_tail), np.mean(avg_f1_tail), med_cnt_tail / visit_cnt_tail,
            )
        )
    if visit_cnt_head > 0:
        llprint(
            "\nHead Performance:\n DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate_head, np.mean(ja_head), np.mean(prauc_head), np.mean(avg_p_head), np.mean(avg_r_head), np.mean(avg_f1_head), med_cnt_head / visit_cnt_head,
            )
        )

    return (
        [ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt],  # 整体性能
        [ddi_rate_tail, np.mean(ja_tail), np.mean(prauc_tail), np.mean(avg_p_tail), np.mean(avg_r_tail), np.mean(avg_f1_tail), med_cnt_tail / visit_cnt_tail if visit_cnt_tail > 0 else 0],  # "tail" 类性能
        [ddi_rate_head, np.mean(ja_head), np.mean(prauc_head), np.mean(avg_p_head), np.mean(avg_r_head), np.mean(avg_f1_head), med_cnt_head / visit_cnt_head if visit_cnt_head > 0 else 0],  # "head" 类性能
    )


def model_test(model_name, model, data_test, voc_size, dataset_path, resume_path, 
               ddi_flag, ddi_adj_path=None, config=None,
               hessian_flag=False, device='cuda:0'):
    model.load_state_dict(torch.load(open(resume_path, "rb")))
    model.to(device=device)
    tic = time.time()
    model.eval()

    # Branch: Calculating the Hessian spectral density
    if hessian_flag:
        results = get_hessian_spectral_density(model_name, model, data_test, voc_size, ddi_flag, 
                                 method='hutchinson', num_samples_per_group=20, num_eigenvalues=50,
                                 ddi_adj_path=ddi_adj_path, config=config, device=device)
    # Branch: Conventional Model Evaluation
    else:
        result_overall = []
        result_tail = []
        result_head = []
        
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )

            overall_metrics, tail_metrics, head_metrics = model_eval(
                model_name, model, test_sample, voc_size, dataset_path, 0)
            result_overall.append(overall_metrics)
            result_tail.append(tail_metrics)
            result_head.append(head_metrics)

        result_overall = np.array(result_overall)
        result_tail = np.array(result_tail)
        result_head = np.array(result_head)

        mean_overall = result_overall.mean(axis=0)
        std_overall = result_overall.std(axis=0)

        mean_tail = result_tail.mean(axis=0)
        std_tail = result_tail.std(axis=0)

        mean_head = result_head.mean(axis=0)
        std_head = result_head.std(axis=0)

        outstring = "Overall: "
        for m, s in zip(mean_overall, std_overall):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        
        outstring = outstring.strip(" & ")
        print(outstring)

        outstring = "Tail: "
        for m, s in zip(mean_tail, std_tail):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        
        outstring = outstring.strip(" & ")

        outstring = "Head: "
        for m, s in zip(mean_head, std_head):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        
        outstring = outstring.strip(" & ")
        print(outstring)

        print("test time: {}".format(time.time() - tic))

# ========================================

def compute_perturbation(model, loss, rho=0.1, device='cuda'):
    grad_w = compute_gradient(model, loss, device, retain_graph=True)

    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad_w])) + 1e-8  # Added epsilon for stability
    perturbation = [rho * g / grad_norm for g in grad_w]
    
    perturbation_tensor = torch.cat([p.view(-1) for p in perturbation])
    perturbation_tensor = torch.clamp(perturbation_tensor, min=-1.0, max=1.0)
    return perturbation_tensor


def generate_random_perturbation(model, rho=0.1, device='cuda'):
    model_params = [param for param in model.parameters()]
    
    perturbation = []
    for param in model_params:
        noise = torch.randn_like(param).to(device)
        perturbation.append(noise * rho)
    
    perturbation_tensor = torch.cat([p.view(-1) for p in perturbation])
    perturbation_tensor = torch.clamp(perturbation_tensor, min=-1.0, max=1.0)
    
    return perturbation_tensor


def compute_gradient(model, loss, device='cuda', retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    gradient = [param.grad.to(device) for param in model.parameters()]
    
    return gradient


def apply_perturbation(model, perturbation, device='cuda'):
    params = [param.data for param in model.parameters()]
    
    all_params = torch.cat([param.view(-1) for param in params], dim=0)
    perturbation = perturbation.view(all_params.size(0))
    updated_params = all_params + perturbation

    offset = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = updated_params[offset:offset + param_size].view(param.size())
        offset += param_size


def remove_perturbation(model, perturbation, device='cuda'):
    params = [param.data for param in model.parameters()]
    
    all_params = torch.cat([param.view(-1) for param in params], dim=0)
    perturbation = perturbation.view(all_params.size(0))
    
    updated_params = all_params - perturbation
    
    offset = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = updated_params[offset:offset + param_size].view(param.size())
        offset += param_size


# ========================================

def get_hessian_spectral_density(model_name, model, data_test, voc_size, ddi_flag, 
                                 method='hutchinson', num_samples_per_group=20, num_eigenvalues=50,
                                 ddi_adj_path=None, config=None, device='cuda:0'):
    model.train()
    if model_name in ["SafeDrug", "GAMENet", "Retain"]:
        module_name = model_name
        function_name = "compute_loss"
        import importlib
        module = importlib.import_module(module_name)
        compute_loss = getattr(module, function_name)

    # 步骤1: 根据adm[4]划分数据
    head_samples = []
    tail_samples = []
    
    for step, input in enumerate(data_test):
        for idx, adm in enumerate(input):
            if len(adm) > 4:
                if adm[4] == "head":
                    head_samples.append((input, idx, adm))
                elif adm[4] == "tail":
                    tail_samples.append((input, idx, adm))
    
    print(f"Head samples: {len(head_samples)}, Tail samples: {len(tail_samples)}")
    
    # 步骤2: 随机采样
    selected_head = random.sample(head_samples, 
                                    min(num_samples_per_group, len(head_samples)))
    selected_tail = random.sample(tail_samples, 
                                    min(num_samples_per_group, len(tail_samples)))
    
    print(f"Selected {len(selected_head)} head samples and {len(selected_tail)} tail samples")
    
    # 步骤3: 为每组计算损失和Hessian谱密度
    results = {}
    
    for group_name, selected_samples in [("head", selected_head), ("tail", selected_tail)]:
        print(f"\n处理 {group_name} 组...")
        
        all_eigenvalues = []
        all_losses = []
        
        for sample_idx, (input, idx, adm) in enumerate(selected_samples):
            print(f"  处理样本 {sample_idx + 1}/{len(selected_samples)}...")
            
            seq_input = input[: idx + 1]
            
            # 计算损失
            with torch.no_grad():
                if model_name in ["SafeDrug"]:
                    loss, _ = compute_loss(
                        model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                        ddi_flag=ddi_flag, device=device, ddi_adj_path=ddi_adj_path, config=config
                    )
                elif model_name in ["GAMENet"]:
                    loss, _, _, _ = compute_loss(
                        model, seq_input, adm, voc_size, 0, 0, ddi_flag=ddi_flag, 
                        device=device, ddi_adj_path=ddi_adj_path, config=config
                    )
                all_losses.append(loss.item())
            
            # 计算Hessian谱密度
            try:
                if method == 'hutchinson':
                    eigenvalues = compute_hessian_spectral_density_hutchinson(
                        model, model_name, compute_loss, seq_input, adm, voc_size, ddi_flag, 
                        device, ddi_adj_path, config, num_samples=num_eigenvalues
                    )
                elif method == 'power_iteration':
                    eigenvalues = compute_hessian_spectral_density_power_iteration(
                        model, model_name, compute_loss, seq_input, adm, voc_size, ddi_flag, 
                        device, ddi_adj_path, config, num_iterations=num_eigenvalues
                    )
                elif method == 'stochastic_lanczos':
                    eigenvalues = compute_hessian_spectral_density_stochastic_lanczos(
                        model, model_name, compute_loss, seq_input, adm, voc_size, ddi_flag, 
                        device, ddi_adj_path, config, num_iterations=30, num_samples=3
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if eigenvalues is not None and len(eigenvalues) > 0:
                    all_eigenvalues.extend(eigenvalues)
                    print(f"    获得 {len(eigenvalues)} 个特征值估计")
                    print(f"    范围: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
            except Exception as e:
                print(f"  样本 {sample_idx + 1} 计算失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 步骤4: 汇总结果
        if len(all_eigenvalues) > 0:
            all_eigenvalues = np.array(all_eigenvalues)
            
            # 计算密度估计
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(all_eigenvalues)
            density_points = np.linspace(all_eigenvalues.min(), 
                                        all_eigenvalues.max(), 500)
            densities = kde(density_points)
            
            # 保存结果
            df = pd.DataFrame({
                'eigenvalue': density_points,
                'density': densities
            })
            
            csv_filename_path = f'{model_name}-hessian_spectral_density'
            temp_path = config['version'] + '/' + config['test_ddi_flag']
            if not os.path.exists(os.path.join(csv_filename_path,  temp_path)):
                os.makedirs(os.path.join(csv_filename_path,  temp_path))

            csv_filename = f'{model_name}-hessian_spectral_density_{group_name}.csv'
            df.to_csv(os.path.join(csv_filename_path,  temp_path, csv_filename), index=False)
            print(f"\n{group_name}组结果已保存到 {csv_filename}")
            
            # 保存原始特征值
            raw_df = pd.DataFrame({
                'eigenvalue': all_eigenvalues
            })
            csv_filename = f'{model_name}-hessian_eigenvalues_raw_{group_name}.csv'
            raw_df.to_csv(os.path.join(csv_filename_path,  temp_path, csv_filename), index=False)
            
            # 输出统计信息
            max_eigenvalue = all_eigenvalues.max()
            min_eigenvalue = all_eigenvalues.min()
            avg_loss = np.mean(all_losses)
            
            print(f"\n{group_name}组统计:")
            print(f"  最大特征值: {max_eigenvalue:.6f}")
            print(f"  最小特征值: {min_eigenvalue:.6f}")
            print(f"  平均损失: {avg_loss:.6f}")
            print(f"  特征值范围: [{min_eigenvalue:.6f}, {max_eigenvalue:.6f}]")
            print(f"  特征值数量: {len(all_eigenvalues)}")
            
            results[group_name] = {
                'max_eigenvalue': max_eigenvalue,
                'min_eigenvalue': min_eigenvalue,
                'avg_loss': avg_loss,
                'eigenvalues': all_eigenvalues,
                'density_points': density_points,
                'densities': densities
            }
        else:
            print(f"{group_name}组: 无法计算特征值")
    
    # 保存汇总统计
    summary_df = pd.DataFrame({
        'group': ['head', 'tail'],
        'max_eigenvalue': [results.get('head', {}).get('max_eigenvalue', np.nan),
                            results.get('tail', {}).get('max_eigenvalue', np.nan)],
        'min_eigenvalue': [results.get('head', {}).get('min_eigenvalue', np.nan),
                            results.get('tail', {}).get('min_eigenvalue', np.nan)],
        'avg_loss': [results.get('head', {}).get('avg_loss', np.nan),
                    results.get('tail', {}).get('avg_loss', np.nan)]
    })
    csv_filename = f'{model_name}-hessian_summary.csv'
    summary_df.to_csv(os.path.join(csv_filename_path,  temp_path, csv_filename), index=False)
    print("\n汇总统计已保存到 hessian_summary.csv")
    
    # print(f"\n总耗时: {time.time() - tic:.2f} 秒")
    
    return results

def compute_hessian_spectral_density_power_iteration(model, model_name, compute_loss_fn, seq_input, adm, 
                                                     voc_size, ddi_flag, device, 
                                                     ddi_adj_path, config,
                                                     num_iterations=50):
    """
    使用幂迭代法估计最大和最小特征值
    更稳定，计算量小
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 保存原始参数
    original_params = [p.data.clone() for p in params]
    
    def compute_loss_and_grad():
        """计算损失和梯度（每次都是新的计算图）"""
        model.zero_grad()
        if model_name in ["SafeDrug"]:
            loss, _ = compute_loss_fn(
                model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                ddi_flag=ddi_flag, device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        elif model_name in ["GAMENet"]:
            loss, _, _, _ = compute_loss_fn(
                model, seq_input, adm, voc_size, 0, 0, ddi_flag=ddi_flag, 
                device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        loss.backward(retain_graph=True)
        
        grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) 
                for p in params]
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        
        return grad_vec
    
    # 计算原始梯度
    original_grad_vec = compute_loss_and_grad()
    
    def hvp(v):
        """Hessian-向量乘积"""
        v_tensor = v.to(device)
        epsilon = 1e-3
        
        # 扰动参数
        offset = 0
        for j, p in enumerate(params):
            numel = p.numel()
            p.data = original_params[j] + epsilon * v_tensor[offset:offset+numel].view_as(p)
            offset += numel
        
        # 重新计算梯度
        perturbed_grad_vec = compute_loss_and_grad()
        
        result = (perturbed_grad_vec - original_grad_vec) / epsilon
        
        # 恢复参数
        for j, p in enumerate(params):
            p.data = original_params[j]
        
        return result
    
    # 估计最大特征值（幂迭代）
    print("    估计最大特征值...")
    v = torch.randn(original_grad_vec.shape[0], device=device)
    v = v / v.norm()
    
    for i in range(num_iterations):
        v = hvp(v)
        eigenvalue = v.norm().item()
        v = v / v.norm()
        if (i + 1) % 10 == 0:
            print(f"      迭代 {i+1}/{num_iterations}, 当前估计: {eigenvalue:.6f}")
    
    max_eigenvalue = eigenvalue
    
    # 估计最小特征值和谱的分布
    print("    估计谱分布...")
    eigenvalues = [max_eigenvalue]
    
    # 使用随机采样估计谱的分布
    for i in range(num_iterations - 1):
        v = torch.randn(original_grad_vec.shape[0], device=device)
        v = v / v.norm()
        Hv = hvp(v)
        rayleigh = (v * Hv).sum().item()
        eigenvalues.append(rayleigh)
        
        if (i + 1) % 10 == 0:
            print(f"      完成 {i+1}/{num_iterations-1}")
    
    # 恢复原始参数
    for j, p in enumerate(params):
        p.data = original_params[j]
    
    return np.array(eigenvalues)


def compute_hessian_spectral_density_hutchinson(model, model_name, compute_loss_fn, seq_input, adm, 
                                                voc_size, ddi_flag, device, 
                                                ddi_adj_path, config,
                                                num_samples=50):
    """
    使用Hutchinson's trace estimator估计Hessian谱密度
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 保存原始参数
    original_params = [p.data.clone() for p in params]
    
    def compute_loss_and_grad():
        """计算损失和梯度"""
        model.zero_grad()
        if model_name in ["SafeDrug"]:
            loss, _ = compute_loss_fn(
                model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                ddi_flag=ddi_flag, device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        elif model_name in ["GAMENet"]:
            loss, _, _, _ = compute_loss_fn(
                model, seq_input, adm, voc_size, 0, 0, ddi_flag=ddi_flag, 
                device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        loss.backward(retain_graph=True)
        
        grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) 
                for p in params]
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        
        return grad_vec
    
    # 计算原始梯度
    original_grad_vec = compute_loss_and_grad()
    
    eigenvalue_estimates = []
    
    print(f"    使用 {num_samples} 个随机向量估计...")
    for i in range(num_samples):
        # 生成随机Rademacher向量
        v = torch.randint(0, 2, (original_grad_vec.shape[0],), device=device).float() * 2 - 1
        
        # 计算Hv（使用有限差分）
        epsilon = 1e-3
        offset = 0
        for j, p in enumerate(params):
            numel = p.numel()
            p.data = original_params[j] + epsilon * v[offset:offset+numel].view_as(p)
            offset += numel
        
        # 重新计算梯度
        perturbed_grad_vec = compute_loss_and_grad()
        
        hvp = (perturbed_grad_vec - original_grad_vec) / epsilon
        
        # v^T H v 是 H 在方向 v 上的Rayleigh商
        rayleigh_quotient = (v * hvp).sum().item()
        eigenvalue_estimates.append(rayleigh_quotient)
        
        # 恢复参数
        for j, p in enumerate(params):
            p.data = original_params[j]
        
        if (i + 1) % 10 == 0:
            print(f"      完成 {i+1}/{num_samples}")
    
    # 恢复原始参数
    for j, p in enumerate(params):
        p.data = original_params[j]
    
    return np.array(eigenvalue_estimates)


def compute_hessian_spectral_density_stochastic_lanczos(model, model_name, compute_loss_fn, seq_input, adm, 
                                                         voc_size, ddi_flag, device, 
                                                         ddi_adj_path, config,
                                                         num_iterations=30, num_samples=3):
    """
    使用随机Lanczos算法估计Hessian谱密度
    这是一个折中方案：比完整Lanczos快，比Hutchinson更准确
    """
    params = [p for p in model.parameters() if p.requires_grad]
    original_params = [p.data.clone() for p in params]
    
    def compute_loss_and_grad():
        """计算损失和梯度"""
        model.zero_grad()
        if model_name in ["SafeDrug"]:
            loss, _ = compute_loss_fn(
                model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                ddi_flag=ddi_flag, device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        elif model_name in ["GAMENet"]:
            loss, _, _, _ = compute_loss_fn(
                model, seq_input, adm, voc_size, 0, 0, ddi_flag=ddi_flag, 
                device=device, ddi_adj_path=ddi_adj_path, config=config
            )
        loss.backward(retain_graph=True)
        
        grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) 
                for p in params]
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        
        return grad_vec
    
    # 计算原始梯度
    original_grad_vec = compute_loss_and_grad()
    
    def hvp(v):
        """Hessian-向量乘积"""
        epsilon = 1e-3
        
        # 扰动参数
        offset = 0
        for j, p in enumerate(params):
            numel = p.numel()
            p.data = original_params[j] + epsilon * v[offset:offset+numel].view_as(p)
            offset += numel
        
        # 重新计算梯度
        perturbed_grad_vec = compute_loss_and_grad()
        
        result = (perturbed_grad_vec - original_grad_vec) / epsilon
        
        # 恢复参数
        for j, p in enumerate(params):
            p.data = original_params[j]
        
        return result
    
    all_eigenvalues = []
    
    print(f"    运行 {num_samples} 次Lanczos迭代...")
    for sample in range(num_samples):
        print(f"      样本 {sample+1}/{num_samples}")
        
        # 随机初始向量
        v = torch.randn(original_grad_vec.shape[0], device=device)
        v = v / v.norm()
        
        # Lanczos迭代
        alpha_list = []
        beta_list = []
        v_prev = torch.zeros_like(v)
        
        for i in range(num_iterations):
            w = hvp(v)
            alpha = (w * v).sum().item()
            alpha_list.append(alpha)
            
            w = w - alpha * v - (beta_list[-1] if beta_list else 0) * v_prev
            beta = w.norm().item()
            
            if beta < 1e-10:
                break
            
            beta_list.append(beta)
            v_prev = v
            v = w / beta
        
        # 构建三对角矩阵
        T = np.diag(alpha_list)
        if len(beta_list) > 0:
            T += np.diag(beta_list, 1) + np.diag(beta_list, -1)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(T)
        all_eigenvalues.extend(eigenvalues)
    
    # 恢复原始参数
    for j, p in enumerate(params):
        p.data = original_params[j]
    
    return np.array(all_eigenvalues)