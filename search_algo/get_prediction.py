# -*- coding: utf-8 -*-
import torch

from settings.config_file import *
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from tqdm import tqdm
import math
import sys
from predictor_models.utils import *
from search_algo.utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from load_data.load_predictor_dataset import *
from search_space_manager.search_space import *
from search_space_manager.sample_models import *
from copy import deepcopy
from predictor_models import *

import importlib


def get_prediction(performance_records_path, e_search_space,predictor_graph_edge_index):
    if (config["predictor"]["predictor_dataset_type"]) == "graph":
        feature_size=train_predictor_using_graph_dataset(performance_records_path)
        TopK_final = predict_and_rank(e_search_space,predictor_graph_edge_index,feature_size)
    elif (config["predictor"]["predictor_dataset_type"]) == "table":
        TopK_final = get_prediction_from_table(performance_records_path, e_search_space)
    return TopK_final


def get_PredictorModel(predictor_model):
    task_model_obj = importlib.import_module(f"predictor_models.{predictor_model}")
    PredictorModel = getattr(task_model_obj, "Predictor")
    train_predictor = getattr(task_model_obj, "train_predictor")
    test_predictor = getattr(task_model_obj, "test_predictor")
    return PredictorModel, train_predictor, test_predictor


def load_predictor_model():
    pass




def train_predictor_using_graph_dataset(predictor_dataset_folder):
    set_seed()
    optimizer = config["predictor"]["optimizer"]
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    lr = float(config["predictor"]["lr"])
    momentum = float(config["predictor"]["momentum"])
    wd = float(config["predictor"]["wd"])
    num_epoch = int(config["predictor"]["num_epoch"])
    start_train_time = time.time()
    train_loader, val_loader, feature_size = load_predictor_dataset(predictor_dataset_folder)
    set_seed()
    predictormodel, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
   
    predictor_model = predictormodel(
        in_channels=feature_size,
        dim=dim,
        drop_out=drop_out,
        out_channels=1)

    predictor_model.to(device)

    optim = map_predictor_optimizer(optimizer)
    optimizer = optim(predictor_model.parameters(),
                      lr=lr,
                      weight_decay=wd)
    criterion = map_predictor_criterion(config["predictor"]["criterion"])

    best_loss = 99999999
    c = 0
    for i in tqdm(range(num_epoch)):
        loss = train_predictor(predictor_model=predictor_model,
                              train_loader=train_loader,
                              criterion=criterion,
                              optimizer=optimizer)

        if loss < best_loss:
            best_loss = loss
            best_predictor_model = copy.deepcopy(predictor_model)

        else:
            c += 1
            if c == int(config["predictor"]['patience']):
                break

    add_config("predictor", "best_loss", round(best_loss,6))
    metrics_list = map_predictor_metrics()
    predictor_performance = test_predictor(model=best_predictor_model,
                                           test_loader=train_loader,
                                           metrics_list=metrics_list,
                                           title="Predictor training test")
    torch.save(best_predictor_model,f"{config['path']['result_folder']}/'predictor_model_weight.pt'")
    for metric, value in predictor_performance.items():
        add_config("results", f"{metric}_train", value)
        print(f"{metric}_train: {value}")
    print(f"Neural predictor training completed in {round((time.time() - start_train_time) / 60, 3)} minutes \n")
    add_config("time", "predictor_training_time", (time.time()-start_train_time))

    predictor_performance = test_predictor(model=best_predictor_model,
                                           test_loader=val_loader,
                                           metrics_list=metrics_list,
                                           title="Predictor validation test")
    for metric, value in predictor_performance.items():
        add_config("results", f"{metric}_val", value)
        print(f"{metric}_test: {value}")

    return feature_size

def predict_and_rank(e_search_space,predictor_graph_edge_index,feature_size):
    set_seed()
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    set_seed()

    # Load predictor model weight
    try:
        dim = int(config["predictor"]["dim"])
        drop_out = float(config["predictor"]["drop_out"])
        model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
        predictor_model = model(
            in_channels=feature_size,
            dim=dim,
            drop_out=drop_out,
            out_channels=1)
        predictor_model = torch.load(f"{config['path']['result_folder']}/'predictor_model_weight.pt'").to(device)
        predictor_model.eval()
    except:
        raise ValueError("Unable to load neural predictor weights, please make sure the neural predictor weight is available and try again ")


    print(f"{'=**=' * 10}  Sampling architecture from search space {'=**=' * 10} ")
    sample_list = random_sampling(e_search_space=e_search_space, n_sample=predict_sample, predictor=True)

    print(f"\n {'=**=' * 10}  Starting predicting architecture performances {'=**=' * 10} ")
    lists = [elt for elt in range(0, len(sample_list), int(config["param"]["batch_sample"]))]
    TopK_models = []
    start_predict_time = time.time()
    for i in tqdm(lists):
        a = i + int(config["param"]["batch_sample"])

        if a > len(sample_list):
            a = len(sample_list)
        sample = sample_list[i:a]

        #    transform model configuration into graph data
        graph_list = []
        for model_config in sample:
            x = get_nodes_features(model_config, e_search_space)
            edge_index = get_edge_index(model_config,predictor_graph_edge_index)
            graphdata = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0],
                             model_config_choices=deepcopy(model_config))
            graph_list.append(graphdata)

        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)

        TopK = predict_neural_performance_using_gnn(predictor_model, sample_dataset)
        TopK_models.append(TopK)

    TopK_model = pd.concat(TopK_models)

    if metric_rule == "max":
         TopK_model = TopK_model.nlargest(k, search_metric, keep="all")
    elif metric_rule == 'min':
        TopK_model = TopK_model.nsmallest(k, search_metric, keep="all")
    else:
        print(
            f"{'++' * 20} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'")
        sys.exit()
    TopK_final = TopK_model[:k]

    prediction_time = round(time.time() - start_predict_time, 2)
    print(f" {len(lists)} Architecture performances predicted in {round(prediction_time / 60, 3)} minutes")
    add_config("time", "pred_time", prediction_time)

    return TopK_final


def predict_neural_performance_using_gnn(model, graphLoader):
    set_seed()
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    model.eval()
    prediction_dict = {'model_config': [], search_metric: []}
    k = int(config["param"]["k"])
    i = 0

    for data in graphLoader:
        performance = []
        i += 1
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        performance = np.append(performance, pred.cpu().detach().numpy())
        choices = deepcopy(data.model_config_choices)
        choice = []
        for a in range(len(pred)):  # loop for retriving the GNN configuration of each graph in the data loader
            temp_list = []
            for key, values in choices.items():
                # temp_list.append((key,values[1][a].item()))
                temp_list.append((key, values[a][1]))
            choice.append(temp_list)
        prediction_dict['model_config'].extend(choice)
        prediction_dict[search_metric].extend(performance)

    df = pd.DataFrame.from_dict(prediction_dict)
    if metric_rule == "max":
         TopK = df.nlargest(n=k, columns=search_metric, keep="all")
    elif metric_rule == 'min':
        TopK = df.nsmallest(n=k, columns=search_metric, keep="all")
    TopK = TopK[:k]
    return TopK


def rank_graphs(model, dataset, batch_size=32):
    set_seed()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    for batch in dataloader:
        with torch.no_grad():
            batch_scores = model(batch)
            scores.extend(batch_scores.cpu().tolist())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_graphs = [(dataset[i], rank+1) for rank, i in enumerate(ranked_indices)]
    return ranked_graphs
