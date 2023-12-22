# -*- coding: utf-8 -*-

# import os

import statistics as stat
from search_space_manager.sample_models import *
from load_data.load_data import load_dataset
from search_space_manager.search_space import *
from search_space_manager.map_functions import *
from search_algo.PCRS import *
from GNN_models.node_classification import *
from GNN_models.graph_classification import *


def get_test_performance(submodel, dataset):
    set_seed()
    z_final= int(config["param"]["z_final"])
    epochs= int(config["param"]["best_model_epochs"])
    timestart = time.time()
    
    print(f'\n Getting final performance  on test dataset')
    train_dataset, val_dataset, test_dataset, in_channels, num_class = load_dataset(dataset)

    print(f"This is the model config:  | {submodel}")
    model_performance = run_model(submodel_config=submodel,
                                  train_data=train_dataset,
                                  test_data=test_dataset,
                                  in_chanels=in_channels,
                                  num_class=num_class,
                                  epochs=epochs,
                                  numround=z_final,
                                  shared_weight=None,
                                  type_data="test",
                                  type_model="final")

    for result, performance in model_performance.items():
        add_config("results", result, model_performance[result])

    return model_performance