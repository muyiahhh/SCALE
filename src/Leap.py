import torch
import argparse
import numpy as np
import dill
import time
from torch.optim import Adam
from tqdm import tqdm
import os
import torch.nn.functional as F
import random
from collections import defaultdict
import yaml
import sys

sys.path.append("..")
from models import Leap
from util import (
    llprint,
    sequence_output_process,
    model_eval,
    model_test,
    get_group_weight,
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

    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)

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

    EPOCH = config['epochs']
    for epoch in tqdm(range(EPOCH), desc="Epochs", ncols=100):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            for adm in input:
                loss = compute_loss(model, adm, END_TOKEN, device)
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

                    perturbed_loss = compute_loss(
                        model=model, adm=adm, END_TOKEN=END_TOKEN, device=device
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
            
            # if step > 10:
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


def fine_tune(fine_tune_name=""):

    # load data
    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"
    device = torch.device("cpu:0")

    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    ddi_A = dill.load(open("../data/output/ddi_A_final.pkl", "rb"))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(
        torch.load(open(os.path.join("saved", args.model_name, fine_tune_name), "rb"))
    )
    model.to(device)

    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=args.lr)
    ddi_rate_record = []

    EPOCH = 100
    for epoch in range(EPOCH):
        loss_record = []
        start_time = time.time()
        # random_train_set = [random.choice(data_train) for _ in range(len(data_train))]
        random.shuffle(data_train)
        for step, input in enumerate(data_train):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(
                    output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1]
                )

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))
                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # llprint("\rtraining step: {} / {}".format(step, len(random_train_set)))

        if K_flag:
            print()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, data_test, voc_size, epoch
            )

    # test
    torch.save(
        model.state_dict(),
        open(os.path.join("saved", args.model_name, "final.model"), "wb"),
    )

def compute_loss(model, adm, END_TOKEN, device='cuda'):
    loss_target = adm[2] + [END_TOKEN]
    output_logits = model(adm)
    loss = F.cross_entropy(
        output_logits, torch.LongTensor(loss_target).to(device)
    )
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafeDrug training/evaluation script')

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config YAML file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
