# -*- coding: utf-8 -*-
import torch

from settings.config_file import *

import sys
from predictor_models.utils import *
from search_algo.utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from load_data.load_predictor_dataset import *
from search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.DDP import *
from copy import deepcopy
from predictor_models import *
import importlib
import torch.optim as optim
import shap
from accelerate import Accelerator
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_prediction_from_table(performance_records_path, e_search_space):
    pass


def get_prediction(performance_records_path, e_search_space, predictor_graph_edge_index, option_decoder):
    if (config["predictor"]["predictor_dataset_type"]) == "graph":
        feature_size,best_predictor_model = train_predictor_using_graph_dataset(performance_records_path)
        if config['param']['search_space_reduction_strategy'] == "none":
            TopK_final = predict_and_rank(e_search_space=e_search_space,
                                          predictor_graph_edge_index=predictor_graph_edge_index,
                                          feature_size=feature_size,
                                          best_predictor_model=best_predictor_model)
        elif config['param']['search_space_reduction_strategy'] == 'probs':
            TopK_final = prob_reduce_and_rank(e_search_space=e_search_space,
                                              predictor_graph_edge_index=predictor_graph_edge_index,
                                              feature_size=feature_size,
                                              option_decoder=option_decoder,
                                              best_predictor_model=best_predictor_model)
        elif config['param']['search_space_reduction_strategy'] in ["gradients", "shapley_values"]:
            TopK_final = feat_coef_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size,
                                                   performance_records_path,best_predictor_model)
        else:
            raise ValueError("Incorrect value for search space reduction strategy")
    elif (config["predictor"]["predictor_dataset_type"]) == "table":
        TopK_final = get_prediction_from_table(performance_records_path, e_search_space)
    return TopK_final


def get_PredictorModel(predictor_model):
    task_model_obj = importlib.import_module(f"predictor_models.{predictor_model}")
    PredictorModel = getattr(task_model_obj, "Predictor")
    train_predictor = getattr(task_model_obj, "train_predictor")
    test_predictor = getattr(task_model_obj, "test_predictor")
    return PredictorModel, train_predictor, test_predictor


def train_predictor_using_graph_dataset(predictor_dataset_folder):
    set_seed()
    optimizer = config["predictor"]["optimizer"]
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    lr = float(config["predictor"]["lr"])
    wd = float(config["predictor"]["wd"])
    num_epoch = int(config["predictor"]["num_epoch"])
    start_train_time = time.time()
    train_data, val_data, feature_size = load_predictor_dataset(predictor_dataset_folder)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    predictor, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
    set_seed()
    predictor_model = predictor(
        in_channels=feature_size,
        dim=dim,
        drop_out=drop_out,
        out_channels=1)
    optim = map_predictor_optimizer(optimizer)
    optimizer = optim(predictor_model.parameters(),
                      lr=lr,
                      weight_decay=wd)
    criterion = map_predictor_criterion(config["predictor"]["criterion"])
    save_path=f"{config['path']['result_folder']}/predictor_model_weight.pt"
    best_predictor_model = ddp_module(accelerator=accelerator,
                                      total_epochs=num_epoch,
                                      model_to_train=predictor_model,
                                      optimizer=optimizer,
                                      train_dataloader=prepare_data_loader(train_data,batch_size=Batch_Size),
                                      criterion=criterion,
                                      model_trainer=train_predictor,
                                      save_path=save_path)

    unwrapped_model = accelerator.unwrap_model(best_predictor_model)
    accelerator.save(unwrapped_model.state_dict(), save_path)
    # add_config("predictor", "best_loss", round(best_loss, 6))
    # add_config("predictor", "best_loss", round(best_loss, 6))
    metrics_list = map_predictor_metrics()
    predictor_performance = test_predictor(accelerator=accelerator,
                                           model=best_predictor_model,
                                           test_loader=accelerator.prepare(prepare_data_loader(train_data)),
                                           metrics_list=metrics_list,
                                           title="Predictor training test")

    for metric, value in predictor_performance.items():
        add_config("predictor", f"{metric}_train", value)
        print(f"{metric}_train: {value}")
    print(f"Neural predictor training completed in {round((time.time() - start_train_time) / 60, 3)} minutes \n")
    add_config("time", "predictor_training_time", (time.time() - start_train_time))

    predictor_performance = test_predictor(accelerator=accelerator,
                                           model=best_predictor_model,
                                           test_loader=accelerator.prepare(prepare_data_loader(val_data)),
                                           metrics_list=metrics_list,
                                           title="Predictor validation test")
    for metric, value in predictor_performance.items():
        add_config("predictor", f"{metric}_val", value)
        print(f"{metric}_val: {value}")

    return feature_size,best_predictor_model


def predict_and_rank(e_search_space, predictor_graph_edge_index, feature_size,best_predictor_model):
    set_seed()
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    set_seed()
    # Load predictor model weight
    # try:
    #     dim = int(config["predictor"]["dim"])
    #     drop_out = float(config["predictor"]["drop_out"])
    #     model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
    #     predictor_model = model(
    #         in_channels=feature_size,
    #         dim=dim,
    #         drop_out=drop_out,
    #         out_channels=1)
    #
    #     model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
    #     predictor_model.load_state_dict(torch.load(model_weigth_path))
    #     predictor_model = predictor_model.to(device)
    #     predictor_model.eval()
    # except:
    #     raise ValueError("Wrong pre-trained predictor weight")

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
            x,_ = get_nodes_features(model_config, e_search_space)
            edge_index = get_edge_index(model_config, predictor_graph_edge_index)
            graphdata = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0],
                             model_config_choices=deepcopy(model_config))
            graph_list.append(graphdata)
        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)
        TopK = predict_neural_performance_using_gnn(best_predictor_model, sample_dataset)
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


def feat_coef_reduce_and_rank(e_search_space, predictor_graph_edge_index,
                              feature_size, predictor_dataset_folder,best_predictor_model):
    set_seed()
    feature_importance_source = config['param']["search_space_reduction_strategy"]
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]

    base_space = deepcopy(e_search_space)
    total_importance = []
    train_loader, val_loader, feature_size = load_predictor_dataset(predictor_dataset_folder)
    pred_Batch_Size = int(config["predictor"]["pred_Batch_Size"])
    train_loader = DataLoader(train_loader, batch_size=pred_Batch_Size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=pred_Batch_Size, shuffle=False)
    criterion = map_predictor_criterion(config["predictor"]["criterion"])
    start_time = time.time()
    for data_loader in [train_loader, val_loader]:
        if feature_importance_source == "gradients":
            feature_importance = compute_gradient_feature_importance(best_predictor_model, data_loader, criterion)
        elif feature_importance_source == "shapley_values":
            feature_importance = compute_shapley_value(best_predictor_model, data_loader, criterion)
            print("feature importance size is:", len(feature_importance))
        else:
            raise ValueError("Wrong value for feature importance computation type")
        total_importance.append(feature_importance)
    if feature_importance_source == "shapley_values":
        grouped_data = list(zip(*feature_importance))
        # Calculate the mean for each group
        fi = [sum(group) / len(group) for group in grouped_data]
        print("This is the final fi", fi)
    else:
        fi = torch.mean(torch.cat(torch.tensor(total_importance), dim=0), dim=0).tolist()
    print(f"Feature importance details is as follows: | # features:{len(fi)}| max:{max(fi)} |Min:{min(fi)} ")
    for fct, value in base_space.items():  # this loop to remove all options that decrease the model performance
        for option, option_details in value.items():
            option_importance = fi[option_details[0]]
            if option_importance <= 0:
                if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                    e_search_space[fct].pop(option)
                else:
                    print(f"The option {option} is no longer present in the search space")
            else:
                # positive_values = [xn for xn in fi if xn >= 0]
                # mean_ = sum(positive_values) / len(positive_values) if positive_values else 0
                mean_ = sum(fi)/len(fi)
                if option_importance <= mean_*2/3:
                    if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                        e_search_space[fct].pop(option)
                        print(f"The option {option} with importance {option_importance} is removed from the search space")
                    else:
                        print(f"The option {option} is no longer present in the search space")
    get_final_sp_details(e_search_space)
    add_config("time", "sp_reduce", round(time.time() - start_time,4 ))
    sample_list = random_sampling(e_search_space=e_search_space, n_sample=0, predictor=True)
    lists = [elt for elt in range(0, len(sample_list), int(config["param"]["batch_sample"]))]
    TopK_models = []
    start_predict_time = time.time()
    print("Start prediction...")
    for i in tqdm(lists, total=len(lists)):
        a = i + int(config["param"]["batch_sample"])
        if a > len(sample_list):
            a = len(sample_list)
        sample = sample_list[i:a]
        #    transform model configuration into graph data
        graph_list = []
        for model_config in sample:
            # print(f"this is the model configuration {model_config}")
            x,_ = get_nodes_features(model_config, e_search_space)
            edge_index = get_edge_index(model_config, predictor_graph_edge_index)
            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
                             model_config_choices=deepcopy(model_config))
            graph_list.append(graphdata)
        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)
        TopK = predict_neural_performance_using_gnn(best_predictor_model, sample_dataset)
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
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)
    TopK_model.to_excel(f"{config['path']['result_folder']}/Topk_model_configs.xlsx")
    return TopK_final




def compute_shapley_value(model, sample_input, num_samples=100):
    start_time = time.time()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import StackingRegressor
    # import xgboost
    dataset_file=f'{config["path"]["result_folder"]}/shapley_dataset.csv'
    search_metric = config["param"]["search_metric"]
    shap.initjs()
    df = pd.read_csv(dataset_file)
    X= df.drop(search_metric,axis=1)
    Y = df[search_metric]
    # X_encoded = pd.get_dummies(X, drop_first=True)

    # X_train, X_test, Y_train, Y_test = train_test_split(X_encoded,Y, test_size=0.2, random_state=num_seed)
    stkr = StackingRegressor(
        estimators=[ ('rfr', RandomForestRegressor())],
        final_estimator=RandomForestRegressor(),
        cv=3
    )
    model = stkr.fit(X, Y)

    explainer = shap.KernelExplainer(model.predict, X)
    shap_value = explainer.shap_values(X)
    shap.summary_plot(shap_value, X, plot_type="bar")

    print(f"Shapley values are as follows: | # samples:{len(shap_value)}")
    print(shap_value)
    print(f"Shapley computation time is {round((time.time() - start_time)/60, 4)} Minutes")

    return shap_value






def compute_gradient_feature_importance(model, sample_dataset, criterion):
    model.eval()
    feature_importance = []
    for data in sample_dataset:
        data.x.requires_grad_(True)
        optimizer = optim.SGD([data.x], lr=0.01)
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y)
        # Backward pass to compute gradients
        optimizer.zero_grad()
        data.x.retain_grad()
        loss.backward()
        feature_importance.append(data.x.grad)

    return torch.cat(feature_importance, dim=0)







def prob_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size, option_decoder, best_predictor_model,ratio_treshold=0.3):
    set_seed()
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    base_space = {}
    rnd = 1
    while base_space != e_search_space:
        TopK_models = []
        c_count = {}
        base_space = deepcopy(e_search_space)
        sample_list = random_sampling(e_search_space=e_search_space, n_sample=predict_sample, predictor=True)

        lists = [elt for elt in range(0, len(sample_list), int(config["param"]["batch_sample"]))]
        print(f"Start Round Number {rnd} of Search space reduction")
        for i in tqdm(lists, total=len(lists)):
            a = i + int(config["param"]["batch_sample"])
            if a > len(sample_list):
                a = len(sample_list)
            sample = sample_list[i:a]

            #    transform model configuration into graph data
            graph_list = []
            for model_config in sample:
                x,_ = get_nodes_features(model_config, e_search_space)
                edge_index = get_edge_index(model_config, predictor_graph_edge_index)
                graphdata = Data(x=x,
                                 edge_index=edge_index,
                                 num_nodes=x.shape[0],
                                 model_config_choices=deepcopy(model_config))
                graph_list.append(graphdata)
            sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)

            TopK = predict_neural_performance_using_gnn(best_predictor_model, sample_dataset)
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
        for id, row in TopK_model.iterrows():  # this loop to aggregate the occurrence of each option in the top-k architectures
            for elt in row["model_config"]:
                if elt[0] in c_count:
                    if option_decoder[elt[1]] in c_count[elt[0]]:
                        c_count[elt[0]][option_decoder[elt[1]]] += 1
                    else:
                        c_count[elt[0]][option_decoder[elt[1]]] = 0
                else:
                    c_count[elt[0]] = {}
                    c_count[elt[0]][option_decoder[elt[1]]] = 0
        for fct, fct_count in c_count.items():  # this loop to remove all option with low ratio in the search space
            for option, option_count in fct_count.items():
                option_ratio = option_count / len(TopK_model)
                if option_ratio < ratio_treshold:
                    if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                        e_search_space[fct].pop(option)
                    else:
                        print(f"The option {option} is no longer present in the search space")
        print(f"The search space after {rnd} is : {e_search_space}")
        rnd += 1
    print(f"The final search space is {e_search_space}")
    sample_list = random_sampling(e_search_space=e_search_space, n_sample=0, predictor=True)
    lists = [elt for elt in range(0, len(sample_list), int(config["param"]["batch_sample"]))]
    TopK_models = []
    start_predict_time = time.time()
    print("Start prediction...")
    edge_index = get_edge_index(model_config, predictor_graph_edge_index)
    for i in tqdm(lists, total=len(lists)):
        a = i + int(config["param"]["batch_sample"])
        if a > len(sample_list):
            a = len(sample_list)
        sample = sample_list[i:a]
        #    transform model configuration into graph data
        graph_list = []
        for model_config in sample:
          
            x,_ = get_nodes_features(model_config, e_search_space)
            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
                             model_config_choices=deepcopy(model_config))
            graph_list.append(graphdata)
        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)
        TopK = predict_neural_performance_using_gnn(best_predictor_model, sample_dataset)
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
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)
    TopK_model.to_excel(f"{config['path']['result_folder']}/Topk_model_configs.xlsx")
    return TopK_final


def predict_neural_performance_using_gnn(model, graphLoader):
    set_seed()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    graphLoader=accelerator.prepare(graphLoader)
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]

    prediction_dict = {'model_config': [], search_metric: []}
    k = int(config["param"]["k"])
    i = 0
    model.eval()
    for data in graphLoader:

        accelerator.print(f"dataloder size is {len(graphLoader)}")
        accelerator.print(f"This is model configuration type is {type(data.model_config_choices)}")
        # accelerator.print(f"This is model configuration {data.model_config_choices}")
        performance = []
        i += 1
        pred = model(data)
        all_pred = accelerator.gather(pred)
        performance = np.append(performance, all_pred.cpu().detach().numpy())

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
    ranked_graphs = [(dataset[i], rank + 1) for rank, i in enumerate(ranked_indices)]
    return ranked_graphs

def get_final_sp_details(sp):
    sp_size, total_choices, t1 = get_sp_details(sp)
    add_config("search_space", "Final_total_function", sp_size)
    add_config("search_space", "final_total_choices", total_choices)
    add_config("search_space", "final_size", t1)
    print(f"The final search space is {sp}")
    print(f" \n {'*' * 5} Reduced Search Details {'*' * 5} \n")
    print(f'# functions: {sp_size} | # options:{total_choices}| # Architectures {t1}')