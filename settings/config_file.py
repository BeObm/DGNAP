# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from configparser import ConfigParser
import os.path as osp
import os
from datetime import datetime

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
num_workers = 8
num_seed = 42
config = ConfigParser()
Batch_Size = 96 * 3

#
RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")


def set_seed(num_seed=num_seed):
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(num_seed)
    torch.cuda.manual_seed_all(num_seed)
    np.random.seed(num_seed)
    random.seed(num_seed)


# =========== First level of  runing configurations  =================>

project_root_dir = os.path.abspath(os.getcwd())


# Second  level of  running configurations
def create_config_file(type_task, dataset_name):
    configs_folder = osp.join(project_root_dir, f'results/{type_task}/{dataset_name}/{RunCode}')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"

    # No neeed to fill dataset information twice
    config["dataset"] = {

        "dataset_name": dataset_name,  # Citeseer,
        'type_task': type_task,  # it could be "graph classification", "link prediction",node classification
        "dataset_root": f"{project_root_dir}/data/{type_task}",
        "shufle_dataset": False
    }

    # fill other configuration information
    config["param"] = {
        "project_dir": project_root_dir,
        'config_filename': config_filename,
        "run_code": RunCode,
        "budget": 80,
        "k": 10,
        "z_sample": 1,  # Number of time  sampled models are trained before we report their performance
        "z_topk": 1,
        "z_final": 2,
        "train_ratio": 0.4,
        "nfcode": 56,  # number of digit for each function code when using embedding method
        "noptioncode": 8,
        "sample_model_epochs": 50,
        "topk_model_epochs": 50,
        "best_model_epochs": 50,
        "patience": 100,
        'search_metric': "roc_auc",  # matthews_corr_coef, balanced_accuracy_score, accuracy_score, roc_auc, auc_pr
        'best_search_metric_rule': "max",  # max
        "encoding_method": "one_hot",  # ={one_hot, embedding,index_embedding}
        "type_sampling": "controlled_stratified_sampling",
        # random_sampling, uniform_sampling, controlled_stratified_sampling

        "feature_size_choice": "total_choices",
        # total_functions total_choices  # for one hot encoding using graph dataset for predictor, use"total choices
        'type_input_graph': "directed",
        "use_paralell": "no",
        "learning_type": "supervised",
        "predict_sample": 500000,
        "batch_sample": 10000
    }

    config["predictor"] = {
        "predictor_dataset_type": "graph",
        "Predictor_model": "GNN_ranking",  # "GNN_ranking","GNN_performance"
        "predictor_metric": "spearman_corr",
        # , ["R2_score", "pearson_corr", "kendall_corr", "spearman_corr"], ["spearman_corr","map_score", "ndcg_score", "kendall_corr", "Top_k_Acc"]
        "pred_Batch_Size": 64,
        "dim": 128,
        "drop_out": 0.3,
        "lr": 0.005,
        "wd": 0.0001,
        "momentum": 0.8,
        "num_epoch": 500,
        "criterion": "MSELoss",  # , [MSELoss,]  [PairwiseLoss, MarginRankingLoss]
        "optimizer": "adam",
        "patience": 50,
        "best_loss":0
    }

    config["time"] = {
        "distribution_time": 00,
        "sampling_time": 00
    }

    with open(config_filename, "w") as file:
        config.write(file)


def add_config(section_, key_, value_, ):
    if section_ not in list(config.sections()):
        config.add_section(section_)
    config[section_][key_] = str(value_)
    filename = config["param"]["config_filename"]
    with open(filename, "w") as conf:
        config.write(conf)


def get_initial_best_performance():
    metric_rule = config["param"]["best_search_metric_rule"]
    if metric_rule == 'max':
        return -99999999
    elif metric_rule == 'min':
        return 99999999
    else:
        print(
            f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'")
        sys.exit()