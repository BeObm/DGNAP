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
    search_metric = config["param"]["search_metric"]
    z_final= int(config["param"]["z_final"])
    epochs= int(config["param"]["best_model_epochs"])
    timestart = time.time()
    
    print(f'\n Getting final performance  on test dataset')
    train_loader, val_loader, test_loader, in_channels, num_class = load_dataset(dataset)
    train_loader = train_loader.to(device)
    test_loader = test_loader.to(device)

    model_performance = run_model(submodel_config=submodel,
                                  train_data=train_loader,
                                  test_data=test_loader,
                                  in_chanels=in_channels,
                                  num_class=num_class,
                                  epochs=epochs,
                                  numround=z_final,
                                  shared_weight=None,
                                  type_data="test")

    for result, performance in model_performance.items():
        add_config("results", result, model_performance[result])

    return model_performance

