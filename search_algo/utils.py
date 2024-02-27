# -*- coding: utf-8 -*-
import random

from settings.config_file import *
import os
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score,auc, recall_score, balanced_accuracy_score, accuracy_score,f1_score
import warnings




def map_downstream_task_metrics():

    if config["dataset"]["type_task"] == "graph_anomaly":
        return [ "F1_macro","roc_auc","auc_pr","F1_score"] #"precision", "recall","F1_score",

    elif config["dataset"]["type_task"] == "graph_classification":
        return ["Accuracy_score", "Balanced_accuracy_score"]

    elif config["dataset"]["type_task"] == "node_classification":
        return ["Accuracy_score", "Balanced_accuracy_score"]


def precision(y_true, y_pred):

    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    return precision.astype(float)


def recall(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    return recall.astype(float)

def F1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def F1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def roc_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc.astype(float)

def auc_pr(y_true, y_pred):

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return  auc_pr

def MCC_score(y_true, y_pred):

    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc


def Balanced_accuracy_score(y_true, y_pred):

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return balanced_acc


def Accuracy_score(y_true, y_pred):

    acc_score = accuracy_score(y_true, y_pred)
    return acc_score


def build_feature(function, option, num_function, e_search_space):  # creer les feature d un graph provenant d un submodel
    set_seed()
    type_encoding = config["param"]["encoding_method"]
    total_choices = int(config["search_space"]["total_choices"])
    total_function = int(config["search_space"]["total_function"])
    max_option = int(config["search_space"]["max_option"])

    if type_encoding == "one_hot":
        d = np.zeros(total_choices, dtype=int)
        # print("option===",option)
        d[option[1]] = 1
    elif type_encoding == "embedding":
        d = option[2]
    elif type_encoding == "index_embedding":
        if config["param"]["feature_size_choice"] == "total_functions":
            d = np.zeros((total_function), dtype=int)

            d[num_function] = list(e_search_space[function]).index(option[0]) + 1
        else:
            d = np.zeros((total_choices), dtype=int)
            d[option[1]] = list(e_search_space[function]).index(option[0]) + 1
    elif type_encoding == "OneHot":
        d = np.zeros((max_option), dtype=int)
        pos = 2

    return d


def get_nodes_features(model_config, e_search_space):
    nodes_features_list = []
    model_config_choices = []
    num_function = 0
    for function, option in model_config.items():
        # print("function===",function)
        feat = build_feature(function, option, num_function, e_search_space)
        num_function += 1
        nodes_features_list.append(feat)
        model_config_choices.append((function, option[1]))
    x = np.array(nodes_features_list)
    x = torch.tensor(x, dtype=torch.float32)
    return x,model_config_choices


def get_edge_index(model_config, predictor_graph_edge_index):
    edge_dict = predictor_graph_edge_index
    node_idx = {}
    idx = 0
    for functions, options in model_config.items():
        node_idx[functions] = idx
        idx += 1
    source = []
    target = []
    edge_index = []
    for function, options in model_config.items():
        # source.append(node_idx[function])
        # target.append(node_idx[function])

        if config['param']['type_input_graph'] == "undirected":
            for function2, options2 in model_config.items():
                source.append(node_idx[function])
                target.append(node_idx[function2])
                # a=node_idx[function]
                # b=node_idx[function2]
                # print(f"Edge between {a} and {b}")
        else:
            for elt in edge_dict[function]:
                if elt in list(model_config.keys()):
                    source.append(node_idx[function])
                    target.append(node_idx[elt])
    edge_index.append(source)
    edge_index.append(target)
    edge_index = np.array(edge_index)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index


def manage_budget():
    budget = int(config["param"]["budget"])
    k = int(config["param"]["k"])
    z_sample = int(config["param"]["z_sample"])

    z_topk = int(config["param"]["z_topk"])
    z_final = int(config["param"]["z_final"])

    n = int((budget - (k * z_topk) - z_final) / z_sample)
    if n <= 0:
        print("Configuration error, Please change budget realated parameters")
        raise SystemExit

    else:
        add_config("param", "n", n)


def seed_worker(worker_id):
    set_seed()
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    return g


def evaluate_model(y_true, y_pred,type_data="val"):
    metrics_list = map_downstream_task_metrics()
    search_metric = config["param"]["search_metric"]
    if search_metric not in metrics_list:
        print(f"{search_metric} is not a metric available for {config['dataset']['type_task']} task")
        exit()
    predictor_performances = {}


    if type_data=="val":
        metric_obj = importlib.import_module(f"search_algo.utils")
        metric_function = getattr(metric_obj, search_metric)
        predictor_performances[search_metric] = metric_function(np.squeeze(y_true), np.squeeze(y_pred))
    else:
        for metric in metrics_list:
            metric_obj = importlib.import_module(f"search_algo.utils")
            metric_function = getattr(metric_obj, metric)
            predictor_performances[metric] = metric_function(np.squeeze(y_true), np.squeeze(y_pred))



    return predictor_performances

def save_search_space_evolution(search_space, rnd):
    file = f"{config['path']['result_folder']}/search_space_reduction_record.txt"
    with open(file,'a') as record:
        record.write(f"\n\n Search space after {rnd} rounds of search space reduction {'*'*5} \n")
        for k,v in search_space.items():
            record.write(f"{k} : {v} \n")


def create_paths():
    # Create here path for recording model performance distribution
    result_folder = osp.join(project_root_dir, f'results/{config["dataset"]["type_task"]}/{config["dataset"]["dataset_name"]}/{RunCode}')
    os.makedirs(result_folder, exist_ok=True)
    add_config("path", "result_folder", result_folder)

    # Create here path for recording details about the result
    result_detail_folder = osp.join(project_root_dir, f'results/result_details/{config["dataset"]["type_task"]}')
    os.makedirs(result_detail_folder, exist_ok=True)
    add_config("path", "result_detail_folder", result_detail_folder)


    # Create here path for saving plots
    plots_folder = osp.join(result_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    add_config("path", "plots_folder", plots_folder)

    # create here path for saving predictor results
    predictor_results_folder = osp.join(result_folder, f"predictor_training_data(#GPU={config['param']['nb_gpu']})")
    os.makedirs(predictor_results_folder, exist_ok=True)
    add_config("path", "predictor_dataset_folder", predictor_results_folder)

    add_config("path", "predictor_weight_path", result_folder)
def get_sp_details(sp):
    total_choices = 0
    t1 = 1
    sp_size = len(sp.keys())
    for k, v in sp.items():
        t1 = t1 * len(v)
        total_choices += len(v)
    return sp_size,total_choices,t1

def get_list_of_choice(sp):
    liste=[]
    for function,choices in sp.items():
        for choice,value in choices.items():
            liste.append(choice)
    return liste
