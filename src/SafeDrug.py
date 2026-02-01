"""SafeDrug training and evaluation script."""

import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
from tqdm import tqdm
import os
import sys
import torch
import time
from models import SafeDrug
from util import (
    llprint, 
    ddi_rate_score,
    buildMPNN,
    get_group_weight,
    model_eval,
    model_test,
    history_update_and_print,
    compute_perturbation,
    compute_gradient,
    apply_perturbation,
    remove_perturbation
)
import torch.nn.functional as F
import yaml

torch.manual_seed(1203)
np.random.seed(2048)


def main(config):
    print("Configuration:\n")
    for key, value in config.items():
        print(f"{key}: {value}")

    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    dataset_path = os.path.join(parent_directory, "data/" + config['dataset_name'] + "/output")
    if config['version'] == "scale":
        result_path = os.path.join(parent_directory, f"results/{config['dataset_name']}/{config['model_name']}/{config['version']}-{config['ddi_flag']}-{config['weight_function']}-{config['beta']}-{config['gamma']}-{config['rho']}")
    else:
        result_path = os.path.join(parent_directory, f"results/{config['dataset_name']}/{config['model_name']}/{config['version']}-{config['ddi_flag']}")
    if not os.path.exists(os.path.join(parent_directory, result_path)):
        os.makedirs(os.path.join(parent_directory, result_path))

    if not config['Test']:
        # Redirect sys.stdout to the log file
        log_file = open(os.path.join(
            result_path, 
            "output_{}_{}.log".format(config['epochs'], config['lr'])), 'w')
        sys.stdout = log_file

    """Main training and evaluation function."""
    print("Loading data...")
    data_path = os.path.join(dataset_path, "records_final.pkl")
    voc_path = os.path.join(dataset_path, "voc_final.pkl")
    ddi_adj_path = os.path.join(dataset_path, "ddi_A_final.pkl")
    ddi_mask_path = os.path.join(dataset_path, "ddi_mask_H.pkl")
    molecule_path = os.path.join(dataset_path, "atc3toSMILES.pkl")

    device = torch.device("cuda:{}".format(config['cuda']))

    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    print(f"Dataset size - Train: {len(data_train)}, Test: {len(data_test)}, Eval: {len(data_eval)}")

    print("Building molecular graph...")

    MPNNSet, N_fingerprint, average_projection = buildMPNN(
        molecule, med_voc.idx2word, 2, device
    )

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    print(f"Vocabulary size - Diag: {voc_size[0]}, Proc: {voc_size[1]}, Med: {voc_size[2]}")

    print("Initializing SafeDrug model...")

    model = SafeDrug(
        voc_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprint,
        average_projection,
        emb_dim=config['dim'],
        device=device,
    )

    if config['Test']:
        model_test(config['model_name'], model, data_test, voc_size, dataset_path, config['resume_path'], 
               config['ddi_flag'], ddi_adj_path, config, True, device)
        return

    # Training process
    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=float(config['lr']))

    if config['version'] == "scale":
        data_train = get_group_weight(
            data_train, function_name=config['weight_function'], hyper_param=config['beta'], merge_type=config['merge_type'])

    history = defaultdict(list)
    best_epoch, best_score = 0, 0

    print("Starting training...")

    EPOCH = config['epochs']
    for epoch in tqdm(range(EPOCH), desc="Epochs", ncols=100):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            loss = 0

            for idx, adm in enumerate(input):
                seq_input = input[: idx + 1]
                loss, _ = compute_loss(
                    model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                    ddi_flag=config['ddi_flag'], device=device, ddi_adj_path=ddi_adj_path, config=config
                )
                optimizer.zero_grad()

                if config['version'] == "original":
                    loss.backward(retain_graph=True)
                    optimizer.step()
                else:
                    if config['version'] == "scale":
                        gap_loss = loss - config['gamma'] * adm[5] * loss
                        grad_w2 = compute_gradient(model, gap_loss, device, True)

                    # Compute perturbation and apply it
                    perturbation = compute_perturbation(model, loss, config['rho'], device)
                    apply_perturbation(model, perturbation, device)

                    perturbed_loss, _ = compute_loss(
                        model=model, seq_input=seq_input, adm=adm, voc_size=voc_size,
                        ddi_flag=config['ddi_flag'], device=device, ddi_adj_path=ddi_adj_path, config=config
                    )
                    remove_perturbation(model, perturbation, device)
                    grad_w1 = compute_gradient(model, perturbed_loss, device, True)
                    torch.nn.utils.clip_grad_norm_(grad_w1, max_norm=1.0)

                    if config['version'] == "sam":
                        gradient = grad_w1
                    elif config['version'] == "scale":
                        gradient = [config['gamma'] * g1 + g2 for g1, g2 in zip(grad_w1, grad_w2)]
                    torch.nn.utils.clip_grad_norm_(gradient, max_norm=1.0)

                    for param, grad in zip(model.parameters(), gradient):
                        param.grad = grad
                        if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                            print("NaN or Inf detected in model parameters after update")
                            break
                    optimizer.step()
            
            # if step > 100:
            #     break
            # llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        overall_metrics, tail_metrics, head_metrics = model_eval(config['model_name'], model, data_eval, voc_size, dataset_path, epoch)
        print(
            "training time: {}, test time: {}".format(
                time.time() - tic, time.time() - tic2
            )
        )

        history = history_update_and_print(history, overall_metrics, tail_metrics, head_metrics, epoch)
        
        if epoch % 1 == 0:
            torch.save(model.state_dict(), 
                    open(os.path.join(
                        result_path, 
                        "Epoch_{}_DDI_{}.model".format(epoch, config['ddi_flag'])
                        ), "wb"), )

        if epoch != 0 and best_score < overall_metrics[1]:
            best_epoch = epoch
            best_score = overall_metrics[1]

        print("best_epoch: {}".format(best_epoch))

    dill.dump(history, 
            open(os.path.join(
                result_path, "history_{}.pkl".format(config['model_name'])
                ), "wb"), )
    
    log_file.close()
    

def compute_loss(model, seq_input, adm, voc_size, ddi_flag="True", device='cuda', ddi_adj_path=None, config=None):
    loss_bce_target = np.zeros((1, voc_size[2]))
    loss_bce_target[:, adm[2]] = 1

    loss_multi_target = np.full((1, voc_size[2]), -1)
    for idx, item in enumerate(adm[2]):
        loss_multi_target[0][idx] = item

    result, loss_ddi = model(seq_input)

    loss_bce = F.binary_cross_entropy_with_logits(
        result, torch.FloatTensor(loss_bce_target).to(device)
    )

    loss_multi = F.multilabel_margin_loss(
        F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
    )

    result = F.sigmoid(result).detach().cpu().numpy()[0]
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    y_label = np.where(result == 1)[0]
    
    current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)

    if ddi_flag=="True":
        phi = min(0, 1 + (config['target_ddi'] - current_ddi_rate) / config['kp'])
        loss = (
            phi * (0.95 * loss_bce + 0.05 * loss_multi)
            + (1 - phi) * loss_ddi
        )
    elif ddi_flag=="False":
        loss = 0.95 * loss_bce + 0.05 * loss_multi
    else:
        if current_ddi_rate <= config['target_ddi']:
            loss = 0.95 * loss_bce + 0.05 * loss_multi
        else:
            phi = min(0, 1 + (config['target_ddi'] - current_ddi_rate) / config['kp'])
            loss = (
                phi * (0.95 * loss_bce + 0.05 * loss_multi)
                + (1 - phi) * loss_ddi
            )
    return loss, loss_ddi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafeDrug training/evaluation script')

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config YAML file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
