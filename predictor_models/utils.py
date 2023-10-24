from settings.config_file import *
import os
import scipy.stats as stats
import importlib
import math
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, matthews_corrcoef, \
    balanced_accuracy_score, accuracy_score
import itertools


def map_predictor_optimizer(optimizers):
    if optimizers == 'adam':
        return torch.optim.Adam
    elif optimizers == 'sgd':
        return torch.optim.SGD

    elif optimizers == 'Adagrad':
        return torch.optim.Adagrad
    elif optimizers == 'LambdaRank':
        return torch.optim.SGD
    elif optimizers == 'sgd':
        return torch.optim.SGD
    elif optimizers == 'sgd':
        return torch.optim.SGD
    else:
        map_function_error(optimizers)


def get_target(data, output):
    target = []
    l=data.shape[0]
    data = data.cpu().detach().numpy().tolist()
    output = output.cpu().detach().numpy().tolist()
    for i in range(len(data)):

        if data[i] >= output[i]:
            target.append(1)
        else:
            target.append(-1)
    return torch.tensor(np.reshape(target, (l, 1)))



def get_pairwise_labels(graphs, labels):
    num_graphs = len(graphs)
    num_pairs = num_graphs * (num_graphs - 1) // 2

    pairwise_graphs = torch.zeros(num_pairs, graphs[0].num_nodes(), graphs[0].num_features())
    pairwise_labels = torch.zeros(num_pairs)

    k = 0
    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            pairwise_graphs[k] = torch.cat([graphs[i], graphs[j]])
            pairwise_labels[k] = (labels[i] > labels[j]).float()
            k += 1

    return pairwise_graphs, pairwise_labels


def map_predictor_criterion(criterion):
    if criterion == 'SmoothL1Loss':
        return ttorch.nn.SmoothL1Loss(reduction='mean', beta=1)
    elif criterion == "MSELoss":
        return torch.nn.MSELoss(reduction='sum')
    elif criterion == "MarginRankingLoss":
        return torch.nn.MarginRankingLoss(margin=0.001)
    elif criterion == "PairwiseLoss":
        return pairwise_loss
    elif criterion == "ListNet":
        return torch.nn.KLDivLoss()
    else:
        map_function_error(criterion)


def ndcg_scores(Y_true, Y_predicted):
    return ndcg_score([Y_true], [Y_predicted])


def map_score(Y_true, Y_predicted):
    return average_precision_score(Y_true, Y_predicted)


def map_score(Y_true, Y_predicted):
    return average_precision_score(Y_true, Y_predicted)


def roc_auc_score(Y_true, Y_predicted):
    return roc_auc_score(Y_true, Y_predicted)


def R2_score(Y_true, Y_predicted):
    return r2_score(Y_true, Y_predicted)


def pearson_corr(Y_true, Y_predicted):
    return stats.pearsonr(np.squeeze(Y_true), np.squeeze(Y_predicted))[0]


def kendall_corr(Y_true, Y_predicted):
    return stats.kendalltau(np.squeeze(Y_true), np.squeeze(Y_predicted))[0]


def spearman_corr(Y_true, Y_predicted):
    return stats.spearmanr(np.squeeze(Y_true), np.squeeze(Y_predicted))[0]


def Top_k_Acc(Y_true, Y_predicted):
    k = int(config["param"]["k"])
    indices = np.argsort(Y_predicted)[-k:]
    return np.sum(Y_true[indices]) / k


def map_predictor_metrics():
    if config["predictor"]["Predictor_model"] == "GNN_performance":
        return ["ndcg_scores","spearman_corr", "kendall_corr", "R2_score", "pearson_corr"]

    if config["predictor"]["Predictor_model"] == "GNN_ranking":
        return ["ndcg_scores", "spearman_corr", "kendall_corr", "Top_k_Acc",  "R2_score"]


def map_function_error(function):
    raise Exception("Error: {} is a wrong value".format(function))


def evaluate_model_predictor(y_true, y_pred, metrics_list, title="Predictor training"):
    search_metric = config["param"]["search_metric"]
    dataset_name = config["dataset"]["dataset_name"]
    predictor_performances = {}

    for metric in metrics_list:
        metric_obj = importlib.import_module(f"predictor_models.utils")
        metric_function = getattr(metric_obj, metric)
        predictor_performances[metric] = round(metric_function(np.squeeze(y_true), np.squeeze(y_pred)), 4)

    predictor_data = {"True": np.squeeze(y_true), "predicted": np.squeeze(y_pred)}
    df = pd.DataFrame.from_dict(predictor_data)
    df.to_excel(f'{config["path"]["result_folder"]}/{title}_{dataset_name}.xlsx')

    if title == "Predictor training test":
        col = "red"
    elif title == "Predictor validation test":
        col = "dodgerblue"
    elif title == "Predictor evaluation test":
        col = "limegreen"
    else:
        col = "red"

    plt.figure(figsize=(8, 8))

    xmin = min(min(y_pred), min(y_true))
    xmax = max(max(y_pred), max(y_true))

    if xmin < 0:
        xmin = 0
    if xmax > 100:
        xmax = 100

    lst = [a for a in range(int(xmin), int(xmax) + 1)]
    plt.plot(lst, lst, color='black', linewidth=0.6)
    plt.scatter(y_true, y_pred, color=col, linewidth=0.8)

    plt.xlabel(f'True {search_metric}', fontsize=28)
    plt.ylabel(f'Predicted {search_metric}', fontsize=28)
    # plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f'{config["path"]["plots_folder"]}/{title}_{dataset_name}.pdf', bbox_inches="tight", dpi=1000)

    return predictor_performances
