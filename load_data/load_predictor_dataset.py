from torch_geometric.loader import DataLoader
from settings.config_file import *
import glob
import torch


def load_predictor_dataset(dataset_source_path):
    set_seed()
    graphlist = []
    pred_Batch_Size = int(config["predictor"]["pred_Batch_Size"])
    for filename in glob.glob(f'{dataset_source_path}/*'):
        data = torch.load(filename)
        data.y = data.y.view(-1, 1)
        graphlist.append(data)

    graph_list = graphlist[:int(config["param"]["n"])]
    val_size = int(len(graph_list) * 20 / 100)

    val_dataset = graph_list[:val_size]
    train_dataset = graph_list[val_size:]
    print(f" Neural Predictor Dataset Description : ( #Graphs:{len(graph_list)} | Feature size:{train_dataset[1].x.shape[1]} | Train:{len(train_dataset)} | Val:{len(val_dataset)})")

    train_loader = DataLoader(train_dataset, batch_size=pred_Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=pred_Batch_Size, shuffle=False)

    feature_size = train_dataset[1].x.shape[1]

    return train_loader,val_loader,feature_size


def inverse_ranking(my_list):
    sorted_list = sorted(my_list)
    ranks = {sorted_list[i]: i+1 for i in range(len(sorted_list))}
    return [len(my_list) - ranks[num] + 1 for num in my_list]

