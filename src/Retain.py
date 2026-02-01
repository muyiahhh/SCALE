import torch
import torch.nn as nn
import numpy as np
import dill
import time
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import yaml
import sys

sys.path.append("..")
from models import Retain
from util import (
    llprint, 
    multi_label_metric, 
    ddi_rate_score,
    get_n_params,
    buildMPNN,
    get_group_weight,
    model_eval,
    model_test,
    print_head_tail_split,
    history_update_and_print,
    compute_perturbation,
    compute_gradient,
    apply_perturbation,
    remove_perturbation
)

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
        result_path = os.path.join(parent_directory, f"results/{config['dataset_name']}/{config['model_name']}/{config['version']}")
    if not os.path.exists(os.path.join(parent_directory, result_path)):
        os.makedirs(os.path.join(parent_directory, result_path))

    log_file = open(os.path.join(
        result_path, 
        "output_{}_{}.log".format(config['epochs'], config['lr'])), 'w')
    sys.stdout = log_file

    """Main training and evaluation function."""
    print("Loading data...")
    data_path = os.path.join(dataset_path, "records_final.pkl")
    voc_path = os.path.join(dataset_path, "voc_final.pkl")
    
    device = torch.device("cuda:{}".format(config['cuda']))

    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Retain(voc_size, device=device)

    if config['Test']:
        model_test(
            config['model_name'], model, data_test, voc_size, dataset_path, 
            config['resume_path'], device)
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
            if len(input) < 2:
                continue

            loss, loss_weight = compute_loss(model, input, voc_size, device)
            optimizer.zero_grad()

            if config['version'] == "original":
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                if config['version'] == "scale":
                    gap_loss = loss - config['gamma'] * loss_weight
                    grad_w2 = compute_gradient(model, gap_loss, device, True)

                # Compute perturbation and apply it
                perturbation = compute_perturbation(model, loss, config['rho'], device)
                apply_perturbation(model, perturbation, device)

                perturbed_loss, _ = compute_loss(model, input, voc_size, device)
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

def compute_loss(model, input, voc_size, device='cuda'):
    loss_ori = 0
    loss_weight = 0
    for idx, adm in enumerate(input):
        if idx+1 < len(input):
            target = np.zeros((1, voc_size[2]))
            target[:, input[idx+1][2]] = 1

            output_logits = model(input[: idx + 1])
            loss_ori += F.binary_cross_entropy_with_logits(
                output_logits, torch.FloatTensor(target).to(device)
            )
            loss_weight += (adm[5] * F.binary_cross_entropy_with_logits(
                output_logits, torch.FloatTensor(target).to(device)
            ))

    return loss_ori, loss_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafeDrug training/evaluation script')

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config YAML file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)