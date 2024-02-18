# -*- coding: utf-8 -*-
import importlib
from settings.config_file import *


def create_e_search_space(search_space_name):
    search_space_package = importlib.import_module("search_space_manager.search_space")
    search_space_function = getattr(search_space_package, f"create_{search_space_name}")
    return search_space_function()


def search_space_embeddings(sp, a=0, b=1):
    set_seed()
    i = 0
    embeddings_dict = {}
    option_decoder = {}  # cle= option code, valeur = option
    fcode_list = []  # list to check duplicate code in function code
    nfcode = int(config["param"]["nfcode"])
    noptioncode = int(config["param"]["noptioncode"])
    total_choices = 0
    t1 = 1
    max_option = 0
    sp_size = len(sp.keys())
    for k, v in sp.items():
        t1 = t1 * len(v)
        total_choices = total_choices + len(v)
        if len(v) > max_option:
            max_option = len(v)
    add_config("search_space", "max_option", max_option)
    add_config("search_space", "total_function", sp_size)
    add_config("search_space", "total_choices", total_choices)
    add_config("search_space", "size_sp", t1)
    add_config("search_space", "Final_total_function", sp_size)
    add_config("search_space", "final_total_choices", total_choices)
    add_config("search_space", "final_size", t1)

    print(f'The search space has {sp_size} Components, a total of {total_choices} choices and {t1} possible GNN models.')

    for function, options_list in sp.items():
        embeddings_dict[function] = {}
        option_code_list = []
        if function in ["gnnConv2", "activation2", "multi_head2", "aggregation2", "normalize2", 'dropout2']:
            for option in options_list:
                option_code = i
                i += 1
                embeddings_dict[function][option] = (option_code, embeddings_dict[f"{function[:-1]}1"][option][1])

                option_decoder[option_code] = option

        else:
            if config["param"]["encoding_method"] == "embedding":
                fcode = [random.randint(a, b) for num in range(0, nfcode)]

                # verifier si une autre fonction na pas le meme code avant de valider le code
                while fcode in fcode_list:
                    fcode = [random.randint(a, b) for num in range(0, nfcode)]
                fcode_list.append(fcode)

                for option in options_list:

                    option_code = i
                    option_encoding = fcode + [random.randint(a, b) for num in range(0, noptioncode)]
                    i += 1
                    while option_encoding in option_code_list:

                        print("option encoding alredy exist")
                        option_encoding = fcode + [random.randint(a, b) for num in range(0, noptioncode)]
                    option_code_list.append(option_encoding)

                    embeddings_dict[function][option] = (option_code, option_encoding)

                    # set decoder dict value for the current option
                    option_decoder[option_code] = option
            else:
                for option in options_list:
                    option_code = i
                    i += 1
                    option_encoding = sp[function].index(option)
                    embeddings_dict[function][option] = (option_code, option_encoding)
                    option_decoder[option_code] = option

    return embeddings_dict, option_decoder


def create_spectral_gnap_gl_space():  # a<b
    sp = {}
    sp['num_graph_filters'] = [2, 3, 4, 5, 8]
    sp['num_signals'] = [1, 2, 3, 4, 6]
    sp['attention'] = ["tanh", "sigmoid", "relu"]  # "Linear"
    sp['aggregation'] = ['add', "max", "mean"]
    sp['normalization'] = [None, 'sym', 'rw']
    sp['activation'] = ["elu", "relu", "sigmoid", "softplus", "tanh"]
    sp['hidden_channels'] = [16, 32, 64, 128]
    sp['dropout'] = [0, 0.2, 0.4, 0.6, 0.8]
    sp['lr_f'] = [0.01, 0.001, 0.0005]
    sp['lr'] = [0.01, 0.001, 0.0005]
    sp['weight_decay'] = [0.001, 0.0001, 0.0005]
    sp["optimizer"] = ["adam","adamW"]
    sp['criterion'] = ['CrossEntropyLoss']  # "MultiMarginLoss"
    sp["graph_filter"] = ["sg", "bernnet", "chebynet", "arma", "appnp"]  # "",,"B-Spline",'appnp',"bernstein",
    e_search_space, option_decoder = search_space_embeddings(sp)

    edge_dict = {}
    edge_dict['gnnConv1'] = ["normalize1", 'dropout1', 'activation1']
    edge_dict['aggregation1'] = ["gnnConv1"]
    edge_dict['multi_head1'] = ["gnnConv1"]
    edge_dict['hidden_channels1'] = ["gnnConv1"]
    edge_dict['normalize1'] = ["dropout1", 'activation1']
    edge_dict['dropout1'] = ["activation1"]
    edge_dict['activation1'] = ["gnnConv2"]
    edge_dict['gnnConv2'] = ["normalize2", 'dropout2', 'activation2']
    edge_dict['aggregation2'] = ["gnnConv2"]
    edge_dict['multi_head2'] = ["gnnConv2"]
    edge_dict['hidden_channels2'] = ["gnnConv2"]
    edge_dict['normalize2'] = ["dropout2", 'activation2']
    edge_dict['dropout2'] = ["activation2"]
    edge_dict['mlp_layer'] = ["graph_filter"]
    edge_dict['hidden_channels'] = ["mlp_layer", "graph_filter"]
    edge_dict['activation'] = ["mlp_layer"]
    edge_dict['graph_filter'] = ["attention"]
    edge_dict['num_graph_filters'] = ["attention"]
    edge_dict['num_signals'] = ["graph_filter"]
    edge_dict['aggregation'] = ["graph_filter"]
    edge_dict['normalization'] = ["graph_filter"]
    edge_dict['lr_f'] = ["graph_filter"]
    edge_dict['attention'] = ["criterion", "dropout"]
    edge_dict['lr'] = ["criterion"]
    edge_dict['weight_decay'] = ["criterion"]
    edge_dict['dropout'] = ["mlp_layer"]
    edge_dict['activation2'] = ["pooling"]
    edge_dict['pooling'] = ['criterion']
    edge_dict['lr'] = ["criterion", "weight_decay"]
    edge_dict['weight_decay'] = ["criterion", "lr"]
    edge_dict["criterion"] = ["optimizer"]
    edge_dict["optimizer"] = []
    return e_search_space, option_decoder, edge_dict


def create_spatial_gnap_gl_space():
    attention = ["GCNConv", "GENConv", "SGConv", "linear", "GraphConv", "GATConv"]
    agregation = ['add', "max", "mean"]
    activation = ["Relu", "Elu", "linear", "Softplus", "sigmoid", "tanh", "relu6","leaky_relu"]
    hidden_channels = [16, 32, 64, 128,256]
    normalizer = [False, "GraphNorm", "InstanceNorm","BatchNorm"]
    dropout = [0.0, 0.2, 0.4, 0.6]
    sp = {'gnnConv1': attention, 'gnnConv2': attention, 'aggregation1': agregation, 'aggregation2': agregation,
          'normalize1': normalizer, 'normalize2': normalizer, 'activation1': activation, 'activation2': activation,
          'hidden_channels': hidden_channels, 'dropout': dropout,
          'lr': [0.1, 0.01, 0.001, 0.005], 'weight_decay': [0, 0.001, 0.0001], "optimizer": ["adam",'adamW'],
          'criterion': ["CrossEntropyLoss"],
          'pooling': ["global_add_pool", "global_max_pool", "global_mean_pool"]}
    e_search_space, option_decoder = search_space_embeddings(sp)

    edge_dict = {'gnnConv1': ["normalize1", 'activation1'], 'aggregation1': ["gnnConv1"],
                 'hidden_channels': ["gnnConv1", "gnnConv2"],
                 'normalize1': ['activation1'], 'activation1': ["gnnConv2"],'activation2': ["pooling"],
                 'gnnConv2': ["normalize2", 'activation2'], 'aggregation2': ["gnnConv2"],
                 'normalize2': ['activation2'],'lr': [],'weight_decay': [],
                 'optimizer': ["weight_decay", "lr", "criterion"], 'dropout': ["pooling"],
                 'pooling': ['criterion'], "criterion": ["optimizer"]}

    return e_search_space, option_decoder, edge_dict


def create_spatial_gnap_nl_space():  # a<b
    attention = ["GCNConv", "GENConv", "SGConv", "linear", "GraphConv", "GATConv"]
    agregation = ['add', "max", "mean"]
    activation = ["Relu", "Elu", "linear", "Softplus", "sigmoid", "tanh", "relu6"]
    multi_head = [1]
    hidden_channels = [16, 64, 128]
    normalizer = ["GraphNorm", "InstanceNorm","BatchNorm"]
    dropout = [0.0, 0.3, 0.5, 0.7]
    sp = {'gnnConv1': attention, 'gnnConv2': attention, 'aggregation1': agregation, 'aggregation2': agregation,
          'normalize1': normalizer, 'normalize2': normalizer, 'activation1': activation, 'activation2': activation,
          'multi_head1': multi_head, 'multi_head2': multi_head, 'hidden_channels1': hidden_channels,
          'hidden_channels2': hidden_channels, 'dropout1': dropout, 'dropout2': dropout,
          'lr': [0.01, 0.001, 0.005, 0.0005], 'weight_decay': [0, 0.001, 0.0005], "optimizer": ["adam","adamW"],
          'criterion': ['CrossEntropyLoss']}

    e_search_space, option_decoder = search_space_embeddings(sp)

    edge_dict = {}
    edge_dict['gnnConv1'] = ["normalize1", 'dropout1', 'activation1']
    edge_dict['aggregation1'] = ["gnnConv1"]
    edge_dict['multi_head1'] = ["gnnConv1"]
    edge_dict['hidden_channels1'] = ["gnnConv1"]
    edge_dict['normalize1'] = ["dropout1", 'activation1']
    edge_dict['dropout1'] = ["activation1"]
    edge_dict['activation1'] = ["gnnConv2"]

    edge_dict['gnnConv2'] = ["normalize2", 'dropout2', 'activation2']
    edge_dict['aggregation2'] = ["gnnConv2"]
    edge_dict['multi_head2'] = ["gnnConv2"]
    edge_dict['hidden_channels2'] = ["gnnConv2"]
    edge_dict['normalize2'] = ["dropout2", 'activation2']
    edge_dict['dropout2'] = ["activation2"]
    edge_dict['mlp_layer'] = ["graph_filter"]
    edge_dict['hidden_channels'] = ["mlp_layer", "graph_filter"]
    edge_dict['activation'] = ["mlp_layer"]
    edge_dict['graph_filter'] = ["attention"]
    edge_dict['num_graph_filters'] = ["attention"]
    edge_dict['num_signals'] = ["graph_filter"]
    edge_dict['aggregation'] = ["graph_filter"]
    edge_dict['normalization'] = ["graph_filter"]
    edge_dict['lr_f'] = ["graph_filter"]
    edge_dict['attention'] = ["criterion", "dropout"]
    edge_dict['lr'] = ["criterion"]
    edge_dict['weight_decay'] = ["criterion"]
    edge_dict['dropout'] = ["mlp_layer"]
    edge_dict['activation2'] = ["criterion"]
    edge_dict['lr'] = ["criterion", "weight_decay"]
    edge_dict['weight_decay'] = ["criterion", "lr"]
    edge_dict["criterion"] = ["optimizer"]
    edge_dict["optimizer"] = []

    return e_search_space, option_decoder, edge_dict

def create_baselines_gl_space():

    attention = ["GCNConv", "GENConv", "linear", "SGConv", 'LEConv', 'ClusterGCNConv', "GATConv"]
    agregation = ['add', "max", "mean"]
    activation = ["elu", "leaky_relu", "linear", "relu", "relu6", "sigmoid", "softplus", "tanh"]
    multi_head = [1, 2, 3, 4]
    hidden_channels = [8, 16, 32, 64, 128]
    dropout = [0.2, 0.4, 0.6, 0.8]
    sp = {'gnnConv1': attention, 'gnnConv2': attention, 'aggregation1': agregation, 'aggregation2': agregation,
          'activation1': activation, 'activation2': activation, 'multi_head1': multi_head, 'multi_head2': multi_head,
          'hidden_channels1': hidden_channels, 'hidden_channels2': hidden_channels, 'dropout1': dropout,
          'dropout2': dropout, 'lr': [1e-2, 1e-3, 1e-4, 5e-3, 5e-4], 'weight_decay': [1e-3, 1e-4, 1e-5, 5e-5, 5e-4],
          'criterion': ["MSELOSS"], 'pooling': ["global_add_pool", "global_mean_pool"], "optimizer": ["adam", "adamW"],
          'normalize1': ["False", "GraphNorm"], 'normalize2': ["False", "GraphNorm"]}

    e_search_space, option_decoder = search_space_embeddings(sp)

    edge_dict = {}
    edge_dict['gnnConv1'] = ["normalize1", 'dropout1', 'activation1']
    edge_dict['aggregation1'] = ["gnnConv1"]
    edge_dict['multi_head1'] = ["gnnConv1"]
    edge_dict['hidden_channels1'] = ["gnnConv1"]
    edge_dict['normalize1'] = ["dropout1", 'activation1']
    edge_dict['dropout1'] = ["activation1"]
    edge_dict['activation1'] = ["gnnConv2"]

    edge_dict['gnnConv2'] = ["normalize2", 'dropout2', 'activation2']
    edge_dict['aggregation2'] = ["gnnConv2"]
    edge_dict['multi_head2'] = ["gnnConv2"]
    edge_dict['hidden_channels2'] = ["gnnConv2"]
    edge_dict['normalize2'] = ["dropout2", 'activation2']
    edge_dict['dropout2'] = ["activation2"]

    edge_dict['mlp_layer'] = ["graph_filter"]
    edge_dict['hidden_channels'] = ["mlp_layer", "graph_filter"]
    edge_dict['activation'] = ["mlp_layer"]
    edge_dict['graph_filter'] = ["attention"]
    edge_dict['num_graph_filters'] = ["attention"]
    edge_dict['num_signals'] = ["graph_filter"]
    edge_dict['aggregation'] = ["graph_filter"]
    edge_dict['normalization'] = ["graph_filter"]
    edge_dict['lr_f'] = ["graph_filter"]
    edge_dict['attention'] = ["criterion", "dropout"]
    edge_dict['lr'] = ["criterion"]
    edge_dict['weight_decay'] = ["criterion"]
    edge_dict['dropout'] = ["mlp_layer"]
    edge_dict['activation2'] = ["pooling"]
    edge_dict['pooling'] = ['criterion']
    edge_dict['lr'] = ["criterion", "weight_decay"]
    edge_dict['weight_decay'] = ["criterion", "lr"]
    edge_dict["criterion"] = ["optimizer"]
    edge_dict["optimizer"] = []

    return e_search_space, option_decoder, edge_dict


def create_baselines_nl_space():

    attention = ["GCNConv", "GENConv", "linear", "SGConv", 'LEConv', 'ClusterGCNConv', "GATConv"]
    agregation = ['add', "max", "mean"]
    activation = ["elu", "leaky_relu", "linear", "relu", "relu6", "sigmoid", "softplus", "tanh"]
    multi_head = [1, 2, 3, 4]
    hidden_channels = [8, 16, 32, 64, 128]
    dropout = [0.2, 0.4, 0.6, 0.8]
    sp = {'gnnConv1': attention, 'gnnConv2': attention, 'aggregation1': agregation, 'aggregation2': agregation,
          'activation1': activation, 'activation2': activation, 'multi_head1': multi_head, 'multi_head2': multi_head,
          'hidden_channels1': hidden_channels, 'hidden_channels2': hidden_channels, 'dropout1': dropout,
          'dropout2': dropout, 'lr': [1e-2, 1e-3, 1e-4, 5e-3, 5e-4], 'weight_decay': [1e-3, 1e-4, 1e-5, 5e-5, 5e-4],
          'criterion': ["fn_loss"], "optimizer": ["adam","adamW"], 'normalize1': ["False"], 'normalize2': ["False"]}

    e_search_space, option_decoder = search_space_embeddings(sp)

    edge_dict = {}
    edge_dict['gnnConv1'] = ["normalize1", 'dropout1', 'activation1']
    edge_dict['aggregation1'] = ["gnnConv1"]
    edge_dict['multi_head1'] = ["gnnConv1"]
    edge_dict['hidden_channels1'] = ["gnnConv1"]
    edge_dict['normalize1'] = ["dropout1", 'activation1']
    edge_dict['dropout1'] = ["activation1"]
    edge_dict['activation1'] = ["gnnConv2"]

    edge_dict['gnnConv2'] = ["normalize2", 'dropout2', 'activation2']
    edge_dict['aggregation2'] = ["gnnConv2"]
    edge_dict['multi_head2'] = ["gnnConv2"]
    edge_dict['hidden_channels2'] = ["gnnConv2"]
    edge_dict['normalize2'] = ["dropout2", 'activation2']
    edge_dict['dropout2'] = ["activation2"]

    edge_dict['mlp_layer'] = ["graph_filter"]
    edge_dict['hidden_channels'] = ["mlp_layer", "graph_filter"]
    edge_dict['activation'] = ["mlp_layer"]
    edge_dict['graph_filter'] = ["attention"]
    edge_dict['num_graph_filters'] = ["attention"]
    edge_dict['num_signals'] = ["graph_filter"]
    edge_dict['aggregation'] = ["graph_filter"]
    edge_dict['normalization'] = ["graph_filter"]
    edge_dict['lr_f'] = ["graph_filter"]
    edge_dict['attention'] = ["criterion", "dropout"]
    edge_dict['lr'] = ["criterion"]
    edge_dict['weight_decay'] = ["criterion"]
    edge_dict['dropout'] = ["mlp_layer"]
    edge_dict['activation2'] = ["criterion"]
    edge_dict['lr'] = ["criterion", "weight_decay"]
    edge_dict['weight_decay'] = ["criterion", "lr"]
    edge_dict["criterion"] = ["optimizer"]
    edge_dict["optimizer"] = []

    return e_search_space, option_decoder, edge_dict
