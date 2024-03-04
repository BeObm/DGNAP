# -*- coding: utf-8 -*-
import torch

from settings.config_file import *
import ast
from predictor_models.utils import *
from search_algo.utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from load_data.load_predictor_dataset import *
# from search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.DDP import *
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
import xgboost
from sklearn.model_selection import train_test_split
# from predictor_models import *
import importlib
import torch.optim as optim
import shap
import pandas as pd
import numpy as np
from predictor_models.utils import get_target
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_prediction_from_table(performance_records_path, e_search_space):
    pass


def get_prediction(performance_records_path, e_search_space, predictor_graph_edge_index, option_decoder):
    if (config["predictor"]["predictor_dataset_type"]) == "graph":
        feature_size,best_predictor_model,accelerator = train_predictor_using_graph_dataset(performance_records_path)
        if config['param']['search_space_reduction_strategy'] == "none":
            TopK_final = predict_and_rank(e_search_space, predictor_graph_edge_index, feature_size)
        elif config['param']['search_space_reduction_strategy'] == 'probs':
            TopK_final = prob_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size, option_decoder)
        elif config['param']['search_space_reduction_strategy'] in ["gradients", "shapley_values"]:
            TopK_final = feat_coef_reduce_and_rank(e_search_space=e_search_space,
                                                   predictor_graph_edge_index=predictor_graph_edge_index,
                                                   predictor_dataset_folder=performance_records_path,
                                                   feature_size=feature_size,
                                                   best_predictor_model=best_predictor_model,
                                                   accelerator=accelerator)
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
    train_data = prepare_predictor_data_loader(train_data, batch_size=Batch_Size, shuffle=True)
    val_data = prepare_predictor_data_loader(val_data, batch_size=Batch_Size, shuffle=False)
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
                                      train_dataloader=train_data,
                                      criterion=criterion,
                                      model_trainer=train_predictor)

    unwrapped_model = accelerator.unwrap_model(best_predictor_model)
    accelerator.save(unwrapped_model.state_dict(), save_path)
    # add_config("predictor", "best_loss", round(best_loss, 6))
    metrics_list = map_predictor_metrics()
    predictor_performance = test_predictor(accelerator=accelerator,
                                           model=best_predictor_model,
                                           test_loader=accelerator.prepare(train_data),
                                           metrics_list=metrics_list,
                                           title="Predictor training test")

    for metric, value in predictor_performance.items():
        add_config("predictor", f"{metric}_train", value)
        print(f"{metric}_train: {value}")
    print(f"Neural predictor training completed in {round((time.time() - start_train_time) / 60, 3)} minutes \n")
    add_config("time", "predictor_training_time", (time.time() - start_train_time))

    predictor_performance = test_predictor(accelerator=accelerator,
                                           model=best_predictor_model,
                                           test_loader=accelerator.prepare(val_data),
                                           metrics_list=metrics_list,
                                           title="Predictor validation test")
    for metric, value in predictor_performance.items():
        add_config("predictor", f"{metric}_val", value)
        print(f"{metric}_val: {value}")

    return feature_size,best_predictor_model,accelerator


def predict_and_rank(e_search_space, predictor_graph_edge_index, feature_size):
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
        model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
        predictor_model.load_state_dict(torch.load(model_weigth_path))
        predictor_model = predictor_model.to(device)
        predictor_model.eval()
    except:
        raise ValueError("Wrong pre-trained predictor weight")
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

def feat_coef_reduce_and_rank(e_search_space, predictor_graph_edge_index,
                              feature_size, predictor_dataset_folder,best_predictor_model,accelerator):
    set_seed()
    feature_importance_source = config['param']["search_space_reduction_strategy"]
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    # Load predictor model and weights

    reduction_start_time = time.time()
    total_importance=[]

    if feature_importance_source == "gradients":
        try:
            model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
            predictor_model = model(
                in_channels=feature_size,
                dim=dim,
                drop_out=drop_out,
                out_channels=1)
            model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
            predictor_model.load_state_dict(torch.load(model_weigth_path))
            predictor_model = predictor_model.to(device)
            predictor_model.eval()
        except:
            raise ValueError("Wrong pre-trained predictor path")
        base_space = deepcopy(e_search_space)
        total_importance = []
        train_loader, _, feature_size = load_predictor_dataset(predictor_dataset_folder, typ="coef")
        pred_Batch_Size = Batch_Size

        criterion = map_predictor_criterion(config["predictor"]["criterion"])
        feature_importance = compute_gradient_feature_importance(predictor_model, train_loader, criterion)
        # print(f"feature importance size is:, |Type: {type(feature_importance)}| Size{feature_importance.shape}")
        # print(" this is feature importance :", feature_importance)
    elif feature_importance_source == "shapley_values":
        feature_importance = compute_shapley_value()# predictor_model, train_loader, criterion
        print("feature importance size is:", len(feature_importance))
    else:
        raise ValueError("Wrong value for feature importance computation type")

    if feature_importance_source == "shapley_values":
        grouped_data = list(zip(*feature_importance))
        # Calculate the mean for each group
        fi = [sum(group) / len(group) for group in grouped_data]
        print("This is the final fi", fi)
    else:
        fi = torch.mean(feature_importance, dim=0).tolist()
    print(f"Feature importance details is as follows: | # features:{len(fi)}| max:{max(fi)} |Min:{min(fi)} ")

    shap_summary=defaultdict(list)
    for fct, value in base_space.items():  # this loop to remove all options that decrease the model performance
        for option, option_details in value.items():
            option_importance = fi[option_details[0]]
            shap_summary["Function"].append(fct)
            shap_summary["Option"].append(option)
            shap_summary["Shap_Value"].append(option_importance)
            if option_importance <= 0:
                if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                    e_search_space[fct].pop(option)
                else:
                    print(f"The option {option} is no longer present in the search space")
            else:
                pass
                # positive_values = [xn for xn in fi if xn >= 0]
                # # mean_ = sum(positive_values) / len(positive_values) if positive_values else 0
                # mean_ = sum(fi)/len(fi)
                # if option_importance <= mean_*2/3:
                #     if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                #         e_search_space[fct].pop(option)
                #         print(f"The option {option} with importance {option_importance} is removed from the search space")
                #     else:
                #         print(f"The option {option} is no longer present in the search space")
    df_sum =pd.DataFrame(shap_summary)
    df_sum.to_excel(f"{config['path']['result_folder']}/Shapley——values-summary.xlsx")
    add_config("time", "sp_reduce", round(time.time() - reduction_start_time, 4))
    get_final_sp_details(e_search_space)
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
        save_gnn_config_file(config['path']["gnn_config_file"],sample)
        for num,model_config in enumerate(sample):
            # print(f"this is the model configuration {model_config}")
            x,_ = get_nodes_features(model_config, e_search_space)
            edge_index = get_edge_index(model_config, predictor_graph_edge_index)
            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
                             model_config_choices=num)
            graph_list.append(graphdata)
        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)
        sample_loader = accelerator.prepare(sample_dataset)
        TopK = predict_neural_performance_using_gnn(accelerator=accelerator,
                                                    model=best_predictor_model,
                                                    graphLoader=sample_loader)
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

def compute_shapley_value():
    num_seed = int(config["param"]["num_seed"])
    nsamples = int(config["param"]["shapley_nsamples"])
    start_time = time.time()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import StackingRegressor
    shap_type = config["param"]["shapley_shap_type"]
    dataset_file=f'{config["path"]["result_folder"]}/shapley_dataset.csv'
    search_metric = config["param"]["search_metric"]
    shap.initjs()
    try:
         df = pd.read_csv(dataset_file)
    except:
        df = pd.read_csv(f"""data/predictor_dataset/{config['dataset']['dataset_name']}/{config['dataset']['dataset_name']}_{config['param']['nb_gpu']}GPU""")
    df = df[:nsamples].sample(frac=1).reset_index(drop=True)
    Xo= df.drop(search_metric,axis=1)
    Yo = df[search_metric]

    # X_encoded = pd.get_dummies(X, drop_first=True)
    X, X_val, Y, Y_val = train_test_split(Xo,Yo, test_size=0.2, random_state=num_seed)
    if shap_type == "kernel":
        stkr = StackingRegressor(
            estimators=[ ('rfr', RandomForestRegressor())],
            final_estimator=RandomForestRegressor(),
            cv=3
        )
        model = stkr.fit(X, Y)

        explainer = shap.KernelExplainer(model.predict, X,algorithm="sampling")
        # explainer = shap.TreeExplainer(model.predict, X)
        shap_value = explainer.shap_values(X)
        shap.summary_plot(shap_value, X, plot_type="bar")
    elif shap_type == "tree":
        stkr = StackingRegressor(
            estimators=[('rfr', RandomForestRegressor())],
            final_estimator=RandomForestRegressor(),
            cv=3
        )
        model = stkr.fit(X, Y)
        # model = xgboost.XGBRegressor(max_depth=1, learning_rate=0.005, subsample=0.5, n_estimators=10000,
        #                                     base_score=Y.mean())
        # model.fit(X, Y, eval_set=[(X_val, Y_val)], verbose=1000)

        # explainer = shap.TreeExplainer(model.predict, X, algorithm="sampling")
        explainer = shap.Explainer(model.predict,X)
        shap_value = explainer.shap_values(X)
        shap.summary_plot(shap_value, X, plot_type="bar",feature_names=list(Xo.columns),title="Shapley Values Summary",)


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
        if (config["predictor"]["Predictor_model"]=="GNN_performance"):
            loss = criterion(output, data.y)
        elif (config["predictor"]["Predictor_model"]=="GNN_ranking"):
            target = get_target(data.y, output).to(device)
            loss = criterion(output, data.y,target)

        # Backward pass to compute gradients
        optimizer.zero_grad()
        data.x.retain_grad()
        loss.backward()
        feature_importance.append(data.x.grad)

    return torch.cat(feature_importance, dim=0)

def prob_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size, option_decoder, ratio_treshold=0.3):
    set_seed()
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    # Load predictor model and weights
    try:
        model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
        predictor_model = model(
             in_channels=feature_size,
             dim=dim,
             drop_out=drop_out,
            out_channels=1)
        model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
        predictor_model.load_state_dict(torch.load(model_weigth_path))
        predictor_model = predictor_model.to(device)
        predictor_model.eval()
    except:
        raise ValueError("Wrong pre-trained predictor path")
    base_space = {}
    rnd = 1
    reduction_start_time = time.time()
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
                option_ratio = (option_count /(len(TopK_model)/len(fct_count)))/len(fct_count)
                if option_ratio < ratio_treshold:
                    if option in e_search_space[fct] and len(e_search_space[fct]) > 1:
                        e_search_space[fct].pop(option)
                    else:
                        print(f"The option {option} is no longer present in the search space")
        print(f"The search space after {rnd} is : {e_search_space}")
        rnd += 1
    add_config("time", "sp_reduce", round(time.time() - reduction_start_time, 4))
    get_final_sp_details(e_search_space)

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
            # print(f"this is the model configuration {model_config}")
            x,_ = get_nodes_features(model_config, e_search_space)
            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
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
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)
    TopK_model.to_excel(f"{config['path']['result_folder']}/Topk_model_configs.xlsx")
    return TopK_final

def gradient_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size, option_decoder, ratio_treshold=0.3):
    set_seed()
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    # Load predictor model and weights
    try:
        model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
        predictor_model = model(
            in_channels=feature_size,
            dim=dim,
            drop_out=drop_out,
            out_channels=1)
        model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
        predictor_model.load_state_dict(torch.load(model_weigth_path))
        predictor_model = predictor_model.to(device)
        predictor_model.eval()
    except:
        raise ValueError("Wrong pre-trained predictor path")
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
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)
    TopK_model.to_excel(f"{config['path']['result_folder']}/Topk_model_configs.xlsx")
    return TopK_final


def prob_reduce_and_rank(e_search_space, predictor_graph_edge_index, feature_size, option_decoder, ratio_treshold=0.3):
    set_seed()
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    # Load predictor model and weights
    try:
        model, train_predictor, test_predictor = get_PredictorModel(config["predictor"]["Predictor_model"])
        predictor_model = model(
            in_channels=feature_size,
            dim=dim,
            drop_out=drop_out,
            out_channels=1)
        model_weigth_path = f"{config['path']['result_folder']}/predictor_model_weight.pt"
        predictor_model.load_state_dict(torch.load(model_weigth_path))
        predictor_model = predictor_model.to(device)
        predictor_model.eval()
    except:
        raise ValueError("Wrong pre-trained predictor path")
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
            # print(f"this is the model configuration {model_config}")
            x,_ = get_nodes_features(model_config, e_search_space)
            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
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
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)
    TopK_model.to_excel(f"{config['path']['result_folder']}/Topk_model_configs.xlsx")
    return TopK_final

@torch.no_grad()
def predict_neural_performance_using_gnn(accelerator, model, graphLoader):
    set_seed()
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    model.eval()
    prediction_dict = {'model_config': [], search_metric: []}
    k = int(config["param"]["k"])
    i = 0
    with open(config["path"]["gnn_config_file"],"r") as cfr_file:
        model_configs = list(cfr_file.readlines())

    for data in graphLoader:
        performance = []
        i += 1
        data = data.to(accelerator.device)
        choice=data.model_config_choices.to(accelerator.device)
        preds = model(data)
        pred = accelerator.gather(preds)
        all_choices=accelerator.gather(choice)
        performance = pred.cpu().detach().numpy()
        choices = all_choices.cpu().detach().numpy()


        records = list(zip(choices,performance))

        for record in records:
            prediction_dict['model_config'].append(retrieve_gnn_config(config["path"]["gnn_config_file"],int(record[0])))
            prediction_dict[search_metric].append(record[1].item())



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