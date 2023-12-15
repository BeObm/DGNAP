# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from search_algo.get_prediction import *
from search_algo.write_results import *
from search_algo.stand_alone import *
from load_data.load_data import *
from search_algo.utils import manage_budget,Generate_time_cost,create_paths
from torch.distributed import init_process_group,destroy_process_group

import time
from settings.config_file import *
import argparse

def setup_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name", default="ENZYMES")
    parser.add_argument("--type_task", help="type_task name", default="graph_classification", choices=["graph_anomaly", "graph_classification", "graph_regression","node_classification"])
    parser.add_argument("--search_space_name", help="search space name", default="spatial_gnap_gl_space")
    parser.add_argument("--sp_reduce", type=str, default="gradients", choices=["none", "probs", "gradients", "shapley_values"],help="search_space_reduction_strategy")
    parser.add_argument("--search_metric", type=str, default="Accuracy_score", help="metric for search guidance")
    parser.add_argument("--predictor", type=str, default="GNN_performance", help="predictor type") # "GNN_ranking","GNN_performance"
    parser.add_argument("--predictor_criterion", type=str, default="MSELoss", help="loss function for predictor")
    parser.add_argument("--world_size", type=str, default=2, help="Number of GPU")
    args = parser.parse_args()
    create_config_file(args.type_task,args.dataset)
    manage_budget()
    add_config("dataset", "dataset_name", args.dataset)
    add_config("param", "search_metric", args.search_metric)
    add_config("param", "search_space_reduction_strategy", args.sp_reduce)
    add_config("predictor", "Predictor_model", args.predictor)
    add_config("predictor", "criterion", args.predictor_criterion)
    add_config("dataset", "type_task", args.type_task)
    create_paths()
    torch.cuda.empty_cache()
    timestart = time.time()
    # torch.cuda.empty_cache()
    type_task = args.type_task
    dataset_name = args.dataset
    dataset_root =config["dataset"]["dataset_root"]
    print(f"{'**'*10} code running on {dataset_name} dataset for {args.type_task} task {'**'*10}")
    
    dataset=get_dataset(type_task,dataset_root,dataset_name)
    # print(dataset)

    e_search_space,option_decoder,predictor_graph_edge_index = create_e_search_space(args.search_space_name)
    performance_records_path = get_performance_distributions(e_search_space=e_search_space,
                                                             dataset=dataset,
                                                             predictor_graph_edge_index=predictor_graph_edge_index,
                                                             world_size=args.world_size)
    # performance_records_path = "data/predictor"
    TopK_final = get_prediction(performance_records_path,e_search_space,predictor_graph_edge_index,option_decoder)
    best_model= get_best_model(TopK_final,option_decoder,dataset,args.world_size)
    total_search_time = round(time.time() - timestart, 2)
    add_config("time", "total_search_time", total_search_time)
    performance = get_test_performance(best_model,dataset)
    write_results(best_model,performance)
    # Generate_time_cost()


