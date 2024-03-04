# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from search_algo.get_prediction import *
from search_algo.write_results import *
from search_algo.stand_alone import *
from load_data.load_data import *
from search_algo.utils import manage_budget,create_paths
import sys

import time
from settings.config_file import *
import argparse

def setup_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name", default="Cora")
    parser.add_argument("--type_task", help="type_task name", default="node_classification", choices=["graph_anomaly", "graph_classification", "graph_regression","node_classification"])
    parser.add_argument("--search_space_name", help="search space name", default="spatial_gnap_nl_space")
    parser.add_argument("--sp_reduce", type=str, default="shapley_values", choices=["none", "probs", "gradients", "shapley_values"],help="search_space_reduction_strategy")
    parser.add_argument("--search_metric", type=str, default="Accuracy_score", help="metric for search guidance")
    parser.add_argument("--best_search_metric_rule", type=str, default="max", help="best search metric rule",choices=["min","max"])
    parser.add_argument("--predictor", type=str, default="GNN_ranking", help="predictor type", choices=["GNN_ranking", "GNN_performance"])
    parser.add_argument("--predictor_criterion", type=str, default="MarginRankingLoss", help="loss function for predictor", choices=["MSELoss","PairwiseLoss", "MarginRankingLoss"])
    parser.add_argument("--nb_gpu", type=str, default=4, help="Number of GPU")
    args = parser.parse_args()
    create_config_file(args.type_task,args.dataset,args.nb_gpu)
    set_seed()
    manage_budget()
    add_config("dataset", "dataset_name", args.dataset)
    add_config("param", "search_metric", args.search_metric)
    add_config("param", "nb_gpu", args.nb_gpu)
    add_config("param", "search_space_reduction_strategy", args.sp_reduce)
    add_config("param", "best_search_metric_rule", args.best_search_metric_rule)
    add_config("predictor", "Predictor_model", args.predictor)
    add_config("predictor", "criterion", args.predictor_criterion)
    add_config("dataset", "type_task", args.type_task)
    add_config("results", f"{args.search_metric}_of_best_sampled_model", 0)
    create_paths()
    timestart = time.time()

    timestart = time.time()
    type_task = args.type_task
    dataset_name = args.dataset
    dataset_root = config["dataset"]["dataset_root"]

    sys.stdout.write(f"{'**' * 10} code running on {dataset_name} dataset for {args.type_task} task {'**' * 10}")

    dataset = get_dataset(type_task, dataset_root, dataset_name)


    e_search_space, option_decoder, predictor_graph_edge_index = create_e_search_space(args.search_space_name)
    # performance_records_path = get_performance_distributions(e_search_space, dataset,predictor_graph_edge_index)
    performance_records_path = f"""data/predictor_dataset/{config['dataset']['dataset_name']}/{config['dataset']['dataset_name']}_{config['param']['nb_gpu']}GPU"""
    search_start = time.time()
    dataset_time_cost = round(search_start - timestart, 2)
    add_config("time", "dataset_time_cost", dataset_time_cost)
    TopK_final = get_prediction(performance_records_path, e_search_space, predictor_graph_edge_index, option_decoder)
    best_model = get_best_model(TopK_final, option_decoder, dataset)
    search_time = round(time.time() - search_start, 2)
    total_search_time = round(time.time() - timestart, 2)
    add_config("time", "search_time", search_time)
    add_config("time", "total_search_time", total_search_time)
    performance={}
    # performance = get_test_performance(best_model, dataset)
    write_results(best_model, performance)
