# -*- coding: utf-8 -*-

from settings.config_file import *
import statistics as stat
from predictor_models.utils import *
from torch_geometric.data import Data
from search_space_manager.sample_models import *
from load_data.load_data import load_dataset
from GNN_models.graph_classification import *

import importlib
from search_algo.DDP import *
from tqdm.auto import tqdm

def get_performance_distributions(e_search_space,
                                  dataset,
                                  predictor_graph_edge_index):  # get performance distribution of s*n models (n = search space size)
    set_seed()
    Batch_Size = int(config['param']['Batch_Size'])
    num_run_sample = int(config["param"]["z_sample"])
    metric_rule = config["param"]["best_search_metric_rule"]
    epochs = int(config["param"]["sample_model_epochs"])
    n_sample = int(config["param"]["N"])
    search_metric = config["param"]["search_metric"]
    list_of_choice=get_list_of_choice(e_search_space)
    timestart = time.time()
    sys.stdout.write(f' \n  {"#" * 10} Getting {search_metric}  of  {n_sample} models {"#" * 10} \n')
    best_performance =get_initial_best_performance()

    model_list = sample_models(n_sample, e_search_space)

    edge_index = get_edge_index(model_list[0], predictor_graph_edge_index)
    predictor_dataset = defaultdict(list)
    shapley_dataset = defaultdict(list)
    graph_list = []

    train_dataset, val_dataset, test_dataset, in_channels, num_class = load_dataset(dataset)
    train_dataset = prepare_data_loader(train_dataset,batch_size=Batch_Size,shuffle=True)
    val_dataset = prepare_data_loader(val_dataset,batch_size=Batch_Size,shuffle=False)

    pbar = tqdm(total=len(model_list))
    pbar.set_description("training samples")
    for no, submodel in tqdm(enumerate(model_list)):

        txt_model = f"Model_Config: {[submodel[opt][0] for opt in submodel.keys()]} "
        submodel_config = {}
        # extract the model config choices
        for key, value in submodel.items():
            submodel_config[key] = value[0]

        model_performance = run_model(
                                      submodel_config=submodel_config,
                                      train_data=train_dataset,
                                      test_data=val_dataset,
                                      in_chanels=in_channels,
                                      num_class=num_class,
                                      epochs=epochs,
                                      numround=num_run_sample,
                                      shared_weight=None,
                                      type_data="val")

        if metric_rule == "max":
            if model_performance > best_performance:
                best_performance = model_performance
                best_sample = copy.deepcopy(submodel)
                best_sample[search_metric] = best_performance

        elif metric_rule == "min":
            if model_performance < best_performance:
                best_performance = model_performance
                best_sample = copy.deepcopy(submodel)
                best_sample[search_metric] = best_performance

        else:
            sys.stdout.write(
                f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'{'++' * 10}")
            sys.exit()

        # =**======**======**======**===  transform model configuration into predictor training sample data ===**======**======**=

        if (config["predictor"]["predictor_dataset_type"]) == "graph":
            x,model_config_choice = get_nodes_features(submodel, e_search_space)
            y = np.array(model_performance)
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            graphdata = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0],
                             model_config_choices=deepcopy(submodel))
            graph_list.append(graphdata)
            torch.save(graphdata, f"{config['path']['predictor_dataset_folder']}/graph{no + 1}_{x.shape[1]}Feats.pt")

            one_hot = np.zeros(len(list_of_choice), dtype=int)
            for elt in model_config_choice:
                one_hot[elt[1] - 1] = 1
            for id, option in enumerate(list_of_choice):
                shapley_dataset[id].append(one_hot[id])
            shapley_dataset[search_metric].append(model_performance)


        elif (config["predictor"]["predictor_dataset_type"]) == "table":
            for function, option in submodel.items():
                if config["param"]["encoding_method"] == "one_hot":
                    predictor_dataset[function].append(option[0])
                elif config["param"]["encoding_method"] == "embedding":
                    predictor_dataset[function].append(option[2])
            predictor_dataset[search_metric].append(model_performance)
        else:
            accelerator.print(
                f"{'++' * 10} Incorrect predictor_dataset_type)")
            sys.exit()
        if accelerator.is_main_process:
            pbar.write(f"{txt_model} | {search_metric}:{round(model_performance,5)}")
            pbar.set_description(f"Training samples.|Best {search_metric}={round(best_performance,5)}")
            pbar.update(1)
    distribution_time = round(time.time() - timestart, 2)
    add_config("time", "distribution_time", distribution_time)
    add_config("results", f"{search_metric}_of_best_sampled_model", best_performance)

    if (config["predictor"]["predictor_dataset_type"]) == "graph":
        df = pd.DataFrame.from_dict(shapley_dataset, orient="columns")
        dataset_file = f'{config["path"]["result_folder"]}/shapley_dataset.csv'
        df.to_csv(dataset_file)
        return config['path']['predictor_dataset_folder']

    if (config["param"]["predictor_dataset_type"]) == "table":
        df = pd.DataFrame.from_dict(predictor_dataset, orient="columns")
        dataset_file = f'{config["path"]["predictor_dataset_folder"]}/{config["dataset"]["dataset_name"]}-{config["param"]["budget"]} samples.csv'
        df.to_csv(dataset_file)
        return dataset_file

def get_best_model(topk_list, option_decoder, dataset):

    set_seed()
    search_metric = config["param"]["search_metric"]
    metric_rule = config["param"]["best_search_metric_rule"]
    best_loss_param_path = f"{config['path']['predictor_weight_path']}/best_dist_param.pth"

    z_topk = int(config["param"]["z_topk"])
    epochs = int(config["param"]["topk_model_epochs"])
    start_time = time.time()
    max_performace =get_initial_best_performance()

    try:  # Recuperer le meilleur model present dans le dataset concu en phase 1
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

    train_dataset, val_dataset, test_dataset, in_channels, num_class = load_dataset(dataset)
    train_dataset = prepare_data_loader(train_dataset, batch_size=Batch_Size, shuffle=True)
    val_dataset = prepare_data_loader(val_dataset, batch_size=Batch_Size, shuffle=False)
    for idx, row in topk_list.iterrows():

        num_model += 1
        dict_model = {}  #
        txt_model = f"Model_Config: {row['model_config']} "
        if (config["predictor"]["predictor_dataset_type"]) == "graph":

            for choice,value in row["model_config"].items():
                dict_model[choice] = value[0]

        elif (config["param"]["predictor_dataset_type"]) == "table":
            for function in topk_list.columns:
                if function != search_metric:
                    if config["param"]["encoding_method"] == "one_hot":
                        dict_model[function] = row[function]
                    elif config["param"]["encoding_method"] == "embedding":
                        dict_model[function] = option_decoder[row[function]]


        model_performance = run_model(submodel_config=dict_model,
                                      train_data=train_dataset,
                                      test_data=val_dataset,
                                      in_chanels=in_channels,
                                      num_class=num_class,
                                      epochs=epochs,
                                      numround=z_topk,
                                      shared_weight=best_loss_param_path,
                                      type_data="val")
        predicted_performance.append(row[search_metric])
        true_performance.append(model_performance)
        sys.stdout.write(f"Architecture {num_model}/{len(topk_list)}:{[dict_model[opt] for opt in dict_model.keys()]} : {search_metric}={model_performance}")

        if metric_rule == "max":
            if model_performance > max_performace:
                max_performace = model_performance
                bestmodel = copy.deepcopy(dict_model)

        elif metric_rule == "min":
            if model_performance < max_performace:
                max_performace = model_performance
                bestmodel = copy.deepcopy(dict_model)

        else:
            sys.stdout.write(
                f"{'++' * 10} {metric_rule} is an invalid rule. Metric rule should be 'min' or 'max'{'++' * 10}")
            sys.exit()

    predictor_performance = evaluate_model_predictor(true_performance, predicted_performance, metrics_list,
                                                     title="Predictor test")
    for metric, value in predictor_performance.items():
        add_config("predictor", f"{metric}_test", value)
    add_config("results", f"Best_model_{search_metric}", round(max_performace,5))
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
            comp_tmp = component
        component_module = importlib.import_module("search_space_manager.map_functions")
        map_func = getattr(component_module, f"map_{comp_tmp}")
        map_func_value = map_func(value)
        model_config[component] = map_func_value
    return model_config


def run_model(submodel_config, train_data, test_data, in_chanels,
              num_class, epochs, numround=1, shared_weight=None, type_data="val",type_model="architecture"):

    search_metric = config["param"]["search_metric"]
    GNN_Model, train_model, test_model = get_train(config["dataset"]["type_task"])
    params_config = get_option_maps(submodel_config)
    params_config["in_channels"] = in_chanels
    params_config["num_class"] = num_class
    set_seed()
    # accelerator.free_memory()
    new_model = GNN_Model(params_config)

    optimizer = params_config["optimizer"](new_model.parameters(),
                                           lr=params_config['lr'],
                                           weight_decay=params_config['weight_decay'])
    if type_model=="final":
         optimizer.param_groups[0]['capturable'] = False
    criterion = params_config["criterion"]
    performance_record = []
    test_performance_record = defaultdict(list)
    for i in range(numround):

        trainer = ddp_module(accelerator=accelerator,
                   total_epochs=epochs,
                   model_to_train=new_model,
                   optimizer=optimizer,
                   train_dataloader=train_data,
                   criterion=criterion,
                   model_trainer=train_model)

        # trainer = (trainer.module if isinstance(trainer, DistributedDataParallel) else trainer)
        trainer.eval()
        performance_score = test_model(model=trainer,
                                       test_loader=accelerator.prepare(test_data),
                                       accelerator=accelerator,
                                       type_data=type_data)
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
