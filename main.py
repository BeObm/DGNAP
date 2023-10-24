# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from  search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.PCRS import *
from search_space_manager.sample_models import *
from search_algo.get_prediction import *
from search_algo.write_results import *
from search_algo.stand_alone import *
from load_data.load_data import *
from search_algo.utils import manage_budget,Generate_time_cost,create_paths
from datetime import date
import random
import time
from settings.config_file import *
import argparse


if __name__ == "__main__":
    set_seed()
    create_config_file()

    manage_budget()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name", default="elliptic")
    parser.add_argument("--search_metric", type=str, default="auc_pr", help="metric for search guidance")
    parser.add_argument("--predictor", type=str, default="GNN_ranking", help="predictor type") #""GNN_ranking","GNN_performance"","GNN_performance"
    parser.add_argument("--criterion", type=str, default="MarginRankingLoss", help="loss function for predictor") #""GNN_ranking","GNN_performance"","GNN_performance"
    args = parser.parse_args()
    add_config("dataset", "dataset_name", args.dataset)
    add_config("param", "search_metric", args.search_metric)
    add_config("predictor", "Predictor_model", args.predictor)
    add_config("predictor", "criterion", args.criterion)
    create_paths()
    torch.cuda.empty_cache()
    

    timestart = time.time()


    # torch.cuda.empty_cache()
    type_task=config["dataset"]["type_task"]
    dataset_name=config["dataset"]["dataset_name"]
    dataset_root =config["dataset"]["dataset_root"]
    print(f"code running on {dataset_name} dataset")
    
    dataset=get_dataset(type_task,dataset_root,dataset_name)
    # print(dataset)

    e_search_space,option_decoder = create_e_search_space()
    # e_search_space,option_decoder = create_baseline_search_space()

    performance_records_path = get_performance_distributions(e_search_space, dataset)
    # performance_records_path = "predictor_training_data_elliptic"
    TopK_final = get_prediction(performance_records_path,e_search_space)
    best_model= get_best_model(TopK_final,option_decoder,dataset)
    total_search_time = round(time.time() - timestart, 2)
    add_config("time", "total_search_time", total_search_time)
    performance = get_test_performance(best_model,dataset)
    write_results(best_model,performance)
    # Generate_time_cost()