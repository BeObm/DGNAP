# -*- coding: utf-8 -*-

import sys
import statistics as stat
from copy import deepcopy
from predictor_models.utils import *
from torch_geometric.data import Data
from search_space_manager.sample_models import *
from load_data.load_data import load_dataset
from GNN_models.graph_classification import *
from settings.config_file import *
import importlib
from search_algo.DDP import *
from tqdm.auto import tqdm

set_seed()

def get_performance_distributions(e_search_space,
                                  dataset,
                                  predictor_graph_edge_index):  # get performance distribution of s*n models (n = search space size)
    set_seed()
    num_run_sample = int(config["param"]["z_sample"])
    metric_rule = config["param"]["best_search_metric_rule"]
    epochs = int(config["param"]["sample_model_epochs"])
    n_sample = int(config["param"]["N"])
    search_metric = config["param"]["search_metric"]

    timestart = time.time()
    print(f' \n  {"#" * 10} Getting {search_metric}  of  {n_sample} models {"#" * 10} \n')

    if metric_rule == 'max':
        best_performance = -99999999
    else:
        best_performance = 99999999
    model_list = sample_models(n_sample, e_search_space)
    edge_index = get_edge_index(model_list[0], predictor_graph_edge_index)
    predictor_dataset = defaultdict(list)
    graph_list = []

    train_dataset, val_dataset, test_dataset, in_channels, num_class = load_dataset(dataset)

    for no, submodel in tqdm(enumerate(model_list)):

        submodel_config = {}
        # extract the model config choices
        for key, value in submodel.items():
            submodel_config[key] = value[0]
        sys.stdout.write(f"Sample {no + 1}/{len(model_list)}: {[submodel[opt][0] for opt in submodel.keys()]} ")

        model_performance = run_model(submodel_config=submodel_config,
                                      train_data=train_dataset,
                                      test_data=val_dataset,
                                      in_chanels=in_channels,
                                      num_class=num_class,
                                      epochs=epochs,
                                      numround=num_run_sample,
                                      shared_weight=None)

        if metric_rule == "max":
            if model_performance > best_performance:
                best_performance = model_performance
                best_sample = copy.deepcopy(submodel)
                best_sample[search_metric] = best_performance
                sys.stdout.write(
                    f" -------> {search_metric} = {round(model_performance, 4)} ===**===> Best Performance \n\n")
            else:
                sys.stdout.write(f" ------> {search_metric} = {round(model_performance, 4)}  \n\n")

        elif metric_rule == "min":
            if model_performance < best_performance:
                best_performance = model_performance
                best_sample = copy.deepcopy(submodel)
                best_sample[search_metric] = best_performance
                sys.stdout.write(
                    f" -------> {search_metric} = {round(model_performance, 4)} ===**===> Best Performance \n\n")
            else:
                sys.stdout.write(f" ------> {search_metric} = {round(model_performance, 4)}  \n\n")
        else:
            print(
                f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'{'++' * 10}")
            sys.exit()

        # =**======**======**======**===  transform model configuration into predictor training sample data ===**======**======**=

        if (config["predictor"]["predictor_dataset_type"]) == "graph":
            x = get_nodes_features(submodel, e_search_space)
            y = np.array(model_performance)
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            graphdata = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0],
                             model_config_choices=deepcopy(submodel))
            graph_list.append(graphdata)
            torch.save(graphdata, f"{config['path']['predictor_dataset_folder']}/graph{no + 1}_{x.shape[1]}Feats.pt")

        elif (config["predictor"]["predictor_dataset_type"]) == "table":
            for function, option in submodel.items():
                if config["param"]["encoding_method"] == "one_hot":
                    predictor_dataset[function].append(option[0])
                elif config["param"]["encoding_method"] == "embedding":
                    predictor_dataset[function].append(option[2])
            predictor_dataset[search_metric].append(model_performance)
        else:
            print(
                f"{'++' * 10} Incorrect predictor_dataset_type)")
            sys.exit()

    distribution_time = round(time.time() - timestart, 2)
    add_config("time", "distribution_time", distribution_time)
    add_config("results", f"best_{search_metric}", best_performance)

    if (config["predictor"]["predictor_dataset_type"]) == "graph":
        return config['path']['predictor_dataset_folder']
    if (config["param"]["predictor_dataset_type"]) == "table":
        df = pd.DataFrame.from_dict(predictor_dataset, orient="columns")
        dataset_file = f'{config["path"]["predictor_dataset_folder"]}/{config["dataset"]["dataset_name"]}-{config["param"]["budget"]} samples.csv'
        df.to_csv(dataset_file)
        return dataset_file


def get_best_model(topk_list, option_decoder, dataset):
    torch.cuda.empty_cache()
    set_seed()
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]

    best_loss_param_path = f"{config['path']['predictor_weight_path']}/best_dist_param.pth"

    z_topk = int(config["param"]["z_topk"])
    epochs = int(config["param"]["topk_model_epochs"])
    start_time = time.time()
    if metric_rule == "max":
        max_performace = -99999999
    elif metric_rule == 'min':
        max_performace = 99999999
    else:
        print(
            f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'")
        sys.exit()
    print(f"\n {'=**=' * 15}  Training top-k models  {'=**=' * 15}")
    try:  # Recuperer le meilleur model present dans le dataset concu en phase 1
        if metric_rule == "max":
            Y = -99999999
        elif metric_rule == 'min':
            Y = 99999999
        for filename in glob.glob(config["path"]["predictor_dataset_folder"] + '/*'):
            data = torch.load(filename)
            data.y = data.y.view(-1, 1)
            if metric_rule == "max":
                if Y < data.y.item():
                    Y = data.y.item()
                    submodel = data.model_config_choices
            elif metric_rule == 'min':
                if Y > data.y.item():
                    Y = data.y.item()
                    submodel = data.model_config_choices
            max_performace = Y
            bestmodel = copy.deepcopy(submodel)
            for k, v in bestmodel.items():
                if k != search_metric:
                    bestmodel[k] = v[0]
    except:
        pass
    num_model = 0
    predicted_performance = []
    true_performance = []
    metrics_list = map_predictor_metrics()
    for idx, row in topk_list.iterrows():
        num_model += 1
        dict_model = {}  #

        if (config["predictor"]["predictor_dataset_type"]) == "graph":
            for choice in row["model_config"]:
                dict_model[choice[0]] = option_decoder[choice[1]]

        elif (config["param"]["predictor_dataset_type"]) == "table":
            for function in topk_list.columns:
                if function != search_metric:
                    if config["param"]["encoding_method"] == "one_hot":
                        dict_model[function] = row[function]
                    elif config["param"]["encoding_method"] == "embedding":
                        dict_model[function] = option_decoder[row[function]]

        train_dataset, val_dataset, test_dataset, in_channels, num_class = load_dataset(dataset)

        sys.stdout.write(f"Architecture {num_model}/{len(topk_list)}:{[dict_model[opt] for opt in dict_model.keys()]} ")
        model_performance = run_model(submodel_config=dict_model,
                                      train_data=train_dataset,
                                      test_data=val_dataset,
                                      in_chanels=in_channels,
                                      num_class=num_class,
                                      epochs=epochs,
                                      numround=z_topk,
                                      shared_weight=best_loss_param_path)
        predicted_performance.append(row[search_metric])
        true_performance.append(model_performance)

        if metric_rule == "max":
            if model_performance > max_performace:
                max_performace = model_performance
                bestmodel = copy.deepcopy(dict_model)
                sys.stdout.write(
                    f" -------> {search_metric} = {round(model_performance, 4)} ===**===> Best Performance \n\n")
            else:
                sys.stdout.write(f" ------> {search_metric} = {round(model_performance, 4)}  \n\n")

        elif metric_rule == "min":
            if model_performance < max_performace:
                max_performace = model_performance
                bestmodel = copy.deepcopy(dict_model)
                sys.stdout.write(
                    f" -------> {search_metric} = {round(model_performance, 4)} ===**===> Best Performance \n\n")
            else:
                sys.stdout.write(f" ------> {search_metric} = {round(model_performance, 4)}  \n\n")
        else:
            print(
                f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'{'++' * 10}")
            sys.exit()

    predictor_performance = evaluate_model_predictor(true_performance, predicted_performance, metrics_list,
                                                     title="Predictor test")
    for metric, value in predictor_performance.items():
        add_config("results", f"{metric}_test", value)
    get_best_model_time = round(time.time() - start_time, 2)
    add_config("time", "get_best_model_time", get_best_model_time)

    return bestmodel


def get_train(type_task):
    task_model_obj = importlib.import_module(f"GNN_models.{type_task}")
    GNN_model = getattr(task_model_obj, "GNN_Model")
    train_model = getattr(task_model_obj, "train_function")
    test_model = getattr(task_model_obj, "test_function")
    return GNN_model, train_model, test_model


def get_option_maps(submodel):
    """
    function to map every component in the search space to its map values
    :param submodel: 
    :return: a dictionary
    """
    model_config = {}
    set_seed()
    for component, value in submodel.items():
        if component in ['gnnConv1', "gnnConv2"]:
            comp_tmp = "gnn_model"
        elif component in ['aggregation1', "aggregation2"]:
            comp_tmp = "aggregation"
        elif component in ['normalize1', "normalize2"]:
            comp_tmp = "normalization"
        elif component in ['activation1', "activation2"]:
            comp_tmp = "activation"
        elif component in ['multi_head1', "multi_head2"]:
            comp_tmp = "multi_head"
        elif component in ['hidden_channels1', "hidden_channels2"]:
            comp_tmp = "hidden_channels"
        elif component in ['dropout1', "dropout2"]:
            comp_tmp = "dropout"
        else:
            comp_tmp=component
        component_module = importlib.import_module("search_space_manager.map_functions")
        map_func = getattr(component_module, f"map_{comp_tmp}")
        map_func_value = map_func(value)
        model_config[component] = map_func_value
    return model_config


def run_model(submodel_config, train_data, test_data, in_chanels, num_class, epochs, numround=1, shared_weight=None,
              type_data="val"):
    set_seed()
    search_metric = config["param"]["search_metric"]
    GNN_Model, train_model, test_model = get_train(config["dataset"]["type_task"])
    params_config = get_option_maps(submodel_config)
    params_config["in_channels"] = in_chanels
    params_config["num_class"] = num_class
    new_model = GNN_Model(params_config)

    if shared_weight != None:
        try:
            new_model.load_state_dict(shared_weight)
        except:
            pass
    optimizer = params_config["optimizer"](new_model.parameters(),
                                           lr=params_config['lr'],
                                           weight_decay=params_config['weight_decay'])
    criterion = params_config["criterion"]
    performance_record = []
    test_performance_record = defaultdict(list)
    c = 0
    for i in range(numround):

            model =ddp_module(
                               total_epochs=epochs,
                               model_to_train=new_model,
                               optimizer=optimizer,
                               train_data=train_data,
                               criterion=criterion,
                               model_trainer=train_model)

            performance_score = test_model(model, test_data, type_data)
            performance_record.append(performance_score[search_metric])

            if type_data == "test":
                for metric, value in performance_score.items():
                    test_performance_record[metric].append(performance_score[metric])

    model_performance = stat.mean(performance_record)
    test_results = {}
    if type_data == "test":
        for metric, value in test_performance_record.items():
            test_results[metric] = round(stat.mean(value), 4)
            test_results[f"{metric}_std"] = round(stat.stdev(value), 4)
        return test_results
    else:
        return model_performance