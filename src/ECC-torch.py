import dill
import numpy as np
import os
import torch
from collections import defaultdict
from tqdm import tqdm
from models import Chain
from util import (
    multi_label_metric,
    ddi_rate_score,
    get_group_weight,
    compute_perturbation,
    generate_random_perturbation,
    compute_gradient,
    apply_perturbation,
    remove_perturbation,
    history_update_and_print
)
import os
import time
import argparse
import yaml
import sys

sys.path.append("..")

def main(config): 
    print("Configuration:\n") 
    for key, value in config.items(): 
        print(f"{key}: {value}") 
 
    current_working_directory = os.getcwd() 
    parent_directory = os.path.dirname(current_working_directory) 
    dataset_path = os.path.join(parent_directory, "data/" + config['dataset_name'] + "/output") 
    result_path = os.path.join(parent_directory, f"results/{config['dataset_name']}/{config['model_name']}/{config['version']}-{config['chain_num']}") 
    if not os.path.exists(os.path.join(parent_directory, result_path)): 
        os.makedirs(os.path.join(parent_directory, result_path)) 
 
    log_file = open(os.path.join(
        result_path, 
        "output_{}_{}_{}.log".format(config['epochs'], config['chain_num'], config['lr'])), 'w')
    sys.stdout = log_file

    """Main training and evaluation function.""" 
    print("Loading data...") 
    data_path = os.path.join(dataset_path, "records_final.pkl") 
    voc_path = os.path.join(dataset_path, "voc_final.pkl") 
 
    # Set device based on config
    device = torch.device("cuda:{}".format(config['cuda']) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data = dill.load(open(data_path, "rb")) 
    voc = dill.load(open(voc_path, "rb")) 
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"] 
 
    epoch = config['epochs'] 
 
    np.random.seed(2048) 
    np.random.shuffle(data) 
    split_point = int(len(data) * 2 / 3) 
    data_train = data[:split_point] 
    eval_len = int(len(data[split_point:]) / 2) 
    data_eval = data[split_point + eval_len :] 
    data_test = data[split_point : split_point + eval_len] 
 
    if config['version'] == "scale": 
        data_train = get_group_weight( 
            data_train, function_name=config['weight_function'], hyper_param=config['beta'], merge_type=config['merge_type']) 
 
    # Create datasets and move to device
    train_X, train_y, train_weight = create_dataset(data_train, diag_voc, pro_voc, med_voc, False) 
    test_X, test_y, split_list = create_dataset(data_test, diag_voc, pro_voc, med_voc, True) 

    appear_idx = np.where(train_y.sum(axis=0) > 0)[0] 
    train_y = train_y[:, appear_idx] 
 
    # Convert data to PyTorch tensors and move to device
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32).to(device) 
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device) 

    train_weight_tensor = torch.tensor(train_weight, dtype=torch.float32).to(device) 
    train_weight_tensor = train_weight_tensor.unsqueeze(1)  # Add a second dimension if necessary 
    train_weight_tensor = train_weight_tensor.expand(-1, train_y_tensor.size(1))  # Broadcast to match output dimension 
 
    input_dim = train_X_tensor.shape[1] 
    length_chain = train_y_tensor.shape[1] 

    tic_total_fit = time.time() 

    history = defaultdict(list)
    best_epoch, best_score = 0, 0

    chains = [Chain(input_dim, 1, length_chain, lr=float(config['lr']), device=device) for _ in range(config['chain_num'])]
    
    if config['Test']:
        chains.load_state_dict(torch.load(open(config['resume_path'], "rb")))
        chains.to(device=device)
        model_test(0, chains, test_X, test_y, split_list, appear_idx, dataset_path, device)
        return

    for epoch in tqdm(range(config['epochs']), desc="Epochs", ncols=100):
        tic = time.time() 
        
        sum_loss = []
        for _, model in enumerate(chains):
            optimizer = model.optimizer
            model.train() 

            # Get output from the i-th classifier
            output = model(train_X_tensor)
            loss = model.criterion(output, train_y_tensor).mean() 
            optimizer.zero_grad() 

            # Compute loss 
            if config['version'] == "original": 
                loss.backward() 
                optimizer.step() 
            else:
                if config['version'] == "scale": 
                    gap_loss = loss - config['gamma'] * (model.criterion(output, train_y_tensor) * train_weight_tensor).mean() 
                    grad_w2 = compute_gradient(model, gap_loss, device, True) 

                # perturbation = compute_perturbation(model, loss, config['rho'], device) 
                perturbation = generate_random_perturbation(model, config['rho'], device)
                apply_perturbation(model, perturbation, device) 

                perturbed_loss = model.criterion(model(train_X_tensor), train_y_tensor).mean() 
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
            
            sum_loss += [loss.item()]

        fittime = time.time() - tic
        print(f"epoch {epoch}, fitting time: {fittime}, average loss: {np.mean(sum_loss)}")

        overall_metrics, tail_metrics, head_metrics = model_test(
            epoch, chains, test_X, test_y, split_list, appear_idx, dataset_path, device)
    
        history = history_update_and_print(history, overall_metrics, tail_metrics, head_metrics, epoch)
    
        if epoch != 0 and best_score < overall_metrics[1]:
            best_epoch = epoch
            best_score = overall_metrics[1]

            if epoch > 100 and epoch % config['eval_interval'] == 0:
                torch.save(model.state_dict(), 
                        open(os.path.join(result_path, "Epoch_{}.model".format(epoch)), "wb"), )

        print("\nbest_epoch: {}".format(best_epoch))

    dill.dump(history, 
            open(os.path.join(
                result_path, "history_{}.pkl".format(config['model_name'])
                ), "wb"), )

    print("total fitting time: {}".format(time.time() - tic_total_fit))
    log_file.close()


def create_dataset(data, diag_voc, pro_voc, med_voc, test_flag=False):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    global output_len
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    weight = []

    if test_flag:
        X_tail, Y_tail, X_head, Y_head = [], [], [], []

    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1
            
            X.append(multi_hot_input)
            y.append(multi_hot_output)
            weight.append(visit[5])

            if test_flag:
                if visit[4] == "tail":
                    X_tail.append(multi_hot_input)
                    Y_tail.append(multi_hot_output)
                elif visit[4] == "head":
                    X_head.append(multi_hot_input)
                    Y_head.append(multi_hot_output)
    
    if test_flag:
        return np.array(X), np.array(y), (np.array(X_tail), np.array(Y_tail), np.array(X_head), np.array(Y_head))
    else:
        return np.array(X), np.array(y), np.array(weight)

def augment(y_pred, appear_idx):
    m, n = y_pred.shape
    y_pred_aug = np.zeros((m, output_len))
    y_pred_aug[:, appear_idx] = y_pred

    return y_pred_aug


def model_inference(chains, test_X_tensor, appear_idx):
    tic = time.time()

    y_pred_chains = []
    y_prob_chains = []

    # Iterate through each model in the chain and perform inference
    for i, model in enumerate(chains):
        model.eval()
        with torch.no_grad():
            y_prob = model(test_X_tensor).cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

            y_pred_chains.append(augment(y_pred, appear_idx))
            y_prob_chains.append(augment(y_prob, appear_idx))
        

    # Convert lists to numpy arrays
    y_pred_chains = np.array(y_pred_chains)
    y_prob_chains = np.array(y_prob_chains)

    pretime = time.time() - tic
    print("Inference time: {}".format(pretime))

    return y_pred_chains, y_prob_chains


def get_pred_prob(y_pred_chains, y_prob_chains):
    y_pred = y_pred_chains.mean(axis=0)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_prob = y_prob_chains.mean(axis=0)
    return y_pred, y_prob


def ddi_rate_score(y_pred, dataset_path):
    ddi_A = dill.load(open(os.path.join(dataset_path, "ddi_A_final.pkl"), "rb"))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    return ddi_rate, med_cnt / visit_cnt


def model_test(epoch, chains, test_X, test_y, split_list, appear_idx, dataset_path, device):
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device) 
    test_tail_X_tensor = torch.tensor(split_list[0], dtype=torch.float32).to(device) 
    test_head_X_tensor = torch.tensor(split_list[2], dtype=torch.float32).to(device) 

    y_pred_chains, y_prob_chains = model_inference(chains, test_X_tensor, appear_idx)
    y_pred, y_prob = get_pred_prob(y_pred_chains, y_prob_chains)
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)
    ddi_rate, medavg = ddi_rate_score(y_pred, dataset_path)

    print("\nEpoch: {} performance on test set:\n".format(epoch))

    print(
        "Overall Performance:\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, medavg
        )
    )

    if test_tail_X_tensor.size(0) > 0:
        y_pred_chains_tail, y_prob_chains_tail = model_inference(chains, test_tail_X_tensor, appear_idx)
        y_pred_tail, y_prob_tail = get_pred_prob(y_pred_chains_tail, y_prob_chains_tail)
        ja_tail, prauc_tail, avg_p_tail, avg_r_tail, avg_f1_tail = multi_label_metric(split_list[1], y_pred_tail, y_prob_tail)
        ddi_rate_score_tail, medavg_tail = ddi_rate_score(y_pred_tail, dataset_path)

        print(
            "Tail Performance:\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate_score_tail, ja_tail, prauc_tail, avg_p_tail, avg_r_tail, avg_f1_tail, medavg_tail
            )
        )
    if test_head_X_tensor.size(0) > 0:
        y_pred_chains_head, y_prob_chains_head = model_inference(chains, test_head_X_tensor, appear_idx)
        y_pred_head, y_prob_head = get_pred_prob(y_pred_chains_head, y_prob_chains_head)
        ja_head, prauc_head, avg_p_head, avg_r_head, avg_f1_head = multi_label_metric(split_list[3], y_pred_head, y_prob_head)
        ddi_rate_score_head, medavg_head = ddi_rate_score(y_pred_head, dataset_path)

        print(
            "Head Performance:\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                ddi_rate_score_head, ja_head, prauc_head, avg_p_head, avg_r_head, avg_f1_head, medavg_head
            )
        )
    
    return (
        [ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), medavg],
        [ddi_rate_score_tail, np.mean(ja_tail), np.mean(prauc_tail), np.mean(avg_p_tail), np.mean(avg_r_tail), np.mean(avg_f1_tail), medavg_tail], 
        [ddi_rate_score_head, np.mean(ja_head), np.mean(prauc_head), np.mean(avg_p_head), np.mean(avg_r_head), np.mean(avg_f1_head), medavg_head], 
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafeDrug training/evaluation script')

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config YAML file")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)