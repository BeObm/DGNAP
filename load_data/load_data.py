import random
import sys

import torch
import torch_geometric
from sklearn.model_selection import StratifiedKFold
from torch import cat
from torch_geometric.data import Data
from load_data.graph_anomaly import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, PPI, Planetoid, Coauthor, Amazon, Flickr, FacebookPagePage
from settings.config_file import *
from scipy.io import loadmat
import pickle
import dgl
import torch.nn.functional as F
import argparse
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random as rd
import numpy as np
import scipy.sparse as sp
import copy as cp

print(f"Torch version: {torch.__version__}")
print(f"Device: {device}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):

        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        set_seed()
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def get_dataset(type_task, dataset_root, dataset_name, normalize_features=True, transform=None):
    set_seed()
    support_dataset_list = {"node_classification": ["Cora", "Citeseer", "Pubmed"],
                            "graph_classification": ["DD", "PROTEINS", "ENZYMES", "BZR", "COLLAB", "IMDB-BINARY"],
                            "graph_anomaly": ["yelp", "elliptic", "Amazon", "YelpChi"]
                            }
    if dataset_name in support_dataset_list[type_task]:

        if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(root=dataset_root, name=dataset_name)  #
            if transform is not None and normalize_features:
                dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
            elif normalize_features == True:
                dataset.transform = T.NormalizeFeatures()
            elif transform is not None:
                dataset.transform = transform

        elif dataset_name in ["DD", "PROTEINS", "ENZYMES", "BZR", "COLLAB", "IMDB-BINARY"]:
            dataset = TUDataset(root=dataset_root, name=dataset_name)  # ,use_node_attr=True,use_edge_attr=True
        elif dataset_name == 'molecule':
            dataset = MoleculeDataset(dataset_root, dataset_name)
            dataset.num_classes = dataset[0].num_classes

        elif dataset_name in ["yelp", "elliptic"]:
            dataset = pickle.load(open(f'{dataset_root}/{dataset_name}.dat', 'rb'))
            dataset.num_classes = 2
        elif dataset_name in ["YelpChi", "Amazon"]:
            dataset = DGLDataset(dataset_name).graph

        else:
            print(f"@@@@@@@@@@  The code for loading  {dataset_name} dataset is not implemented")
            sys.exit()
    else:
        print(f"@@@@@@@@@@  The {dataset_name} dataset is not supported for {type_task} in the current version ")
        sys.exit()

    return dataset


def load_dataset(dataset, batch_dim=Batch_Size):
    type_task = config["dataset"]["type_task"]
    set_seed()

    if type_task == "node_classification":
        if config["param"]["learning_type"] == "supervised":
            test_loader = train_loader = val_loader = Load_nc_data(dataset[0])
        else:
            test_loader = train_loader = val_loader = dataset[0]
        in_channels = dataset[0].x.shape[1]
        num_class = dataset.num_classes
    elif type_task == "graph_anomaly":
        if config['dataset']['dataset_name'] in ["elliptic", "yelp"]:
            test_loader = train_loader = val_loader = dataset
            in_channels = dataset.num_features
        elif config['dataset']['dataset_name'] in ["YelpChi", "Amazon"]:
            # test_loader = train_loader = val_loader = dataset
            test_loader = train_loader = val_loader = Load_autogad_dataset(dataset)
            in_channels = dataset.ndata['feature'].shape[1]
        num_class = 2
    else:
        if config["dataset"]["shufle_dataset"] == True:
            train_dataset, train_dataset, train_dataset = shuffle_dataset(dataset)
        else:
            n = int(len(dataset) * 20 / 100)
            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]

        train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_dim, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)
        add_config("dataset", "len_traindata", len(train_dataset))
        add_config("dataset", "len_testdata", len(test_dataset))
        add_config("dataset", "len_valdata", len(val_loader))

    return train_loader, val_loader, test_loader, in_channels, num_class


def shuffle_dataset(dataset):
    set_seed()
    dataset = ShuffleDataset(dataset, 1024)
    length = 0
    for d in dataset:
        length = length + 1
    n = int(length * 20 / 100)
    tmp_list = [a for a in range(length)]
    random.shuffle(tmp_list)

    test_dataset_idx = tmp_list[0:n]
    val_dataset_idx = tmp_list[n:2 * n]
    train_dataset_idx = tmp_list[2 * n:]

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for i, d in enumerate(dataset):
        if i in train_dataset_idx:
            train_dataset.append(d)
        elif i in val_dataset_idx:
            train_dataset.append(d)
        elif i in test_dataset_idx:
            train_dataset.append(d)
        else:
            continue
    return train_dataset, train_dataset, train_dataset


def Load_nc_data(data, shuffle=True):
    set_seed()
    if shuffle:
        indices = torch.randperm(data.x.size(0))
        data.train_mask = index_to_mask(indices[1500:3500], size=data.num_nodes)
        data.val_mask = index_to_mask(indices[1000:1500], size=data.num_nodes)
        data.test_mask = index_to_mask(indices[:1000], size=data.num_nodes)
    else:
        # data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.train_mask[:1000] = 1
        # data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.val_mask[1000: 1500] = 1
        # data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        # data.test_mask[1500:2000] = 1

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.train_mask[:-1000] = 1
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.val_mask[-1000: -500] = 1
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
        data.test_mask[-500:] = 1
    return data


def get_fraud_edge_index():
    set_seed()
    data = loadmat(f'{config["dataset"]["dataset_root"]}/{config["dataset"]["dataset_name"]}.mat')
    if config["dataset"]["dataset_name"] == "YelpChi":
        # data_rur = data['net_rur']
        # data_rtr = data['net_rtr']
        # data_rsr = data['net_rsr']
        data_homo = data['homo']
        # sparse_to_adjlist(data_rur, config["dataset"]["dataset_root"] + 'yelp_rur_adjlists.pickle')
        # sparse_to_adjlist(data_rtr, config["dataset"]["dataset_root"] + 'yelp_rtr_adjlists.pickle')
        # sparse_to_adjlist(data_rsr, config["dataset"]["dataset_root"] + 'yelp_rsr_adjlists.pickle')
        sparse_to_adjlist(data_homo, config["dataset"]["dataset_root"] + 'yelp_homo_adjlists.pickle')
        with open(config["dataset"]["dataset_root"] + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo_edge_list = pickle.load(file)
            source = []
            target = []
            edge_index = []
            for k, v in homo_edge_list.items():
                for elt in list(v):
                    source.append(k)
                    target.append(elt)

            edge_index.append(source)
            edge_index.append(target)
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index

    elif config["dataset"]["dataset_name"] == "Amazon":
        data = loadmat(f'{config["dataset"]["dataset_root"]}/{config["dataset"]["dataset_name"]}.mat')
        # data_upu = data['net_upu']
        # data_usu = data['net_usu']
        # data_uvu = data['net_uvu']
        data_homo = data['homo']
        # sparse_to_adjlist(data_upu, config["dataset"]["dataset_root"] + 'amz_upu_adjlists.pickle')
        # sparse_to_adjlist(data_usu, config["dataset"]["dataset_root"] + 'amz_usu_adjlists.pickle')
        # sparse_to_adjlist(data_uvu, config["dataset"]["dataset_root"] + 'amz_uvu_adjlists.pickle')
        sparse_to_adjlist(data_homo, config["dataset"]["dataset_root"] + 'amazon_homo_adjlists.pickle')

        with open(config["dataset"]["dataset_root"] + 'amazon_homo_adjlists.pickle', 'rb') as file:
            homo_edge_list = pickle.load(file)

            source = []
            target = []
            edge_index = []
            for k, v in homo_edge_list.items():
                for elt in list(v):
                    source.append(k)
                    target.append(elt)

            edge_index.append(source)
            edge_index.append(target)
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        return edge_index


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    set_seed()
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)

    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def Load_autogad_dataset(g):
    set_seed()
    # features = g.ndata['feature']
    # labels = g.ndata['label'].view(-1, 1)
    data_file = loadmat(f"{config['dataset']['dataset_root']}/{config['dataset']['dataset_name']}.mat")
    labels = data_file['label'].flatten()
    features = data_file['features'].todense().A
    features = torch.tensor(features, dtype=torch.float32)

    edge_index = get_fraud_edge_index()

    if config['dataset']['dataset_name'] == 'YelpChi':
        index = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels,
                                                                stratify=labels,
                                                                test_size=1 - float(config["param"]["train_ratio"]),
                                                                random_state=2,
                                                                shuffle=True)
    elif config['dataset']['dataset_name'] == 'Amazon':  # amazon
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:],
                                                                stratify=labels[3305:],
                                                                test_size=1 - float(config["param"]["train_ratio"]),
                                                                random_state=2, shuffle=True)

    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)

    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    labels = torch.tensor(labels)
    dataset = Data(x=features,
                   edge_index=edge_index,
                   y=labels,
                   num_nodes=len(labels),
                   train_mask=train_mask,
                   val_mask=val_mask,
                   test_mask=test_mask)

    print('train/val/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    return dataset


def Load_nc_data2(data):
    set_seed()

    data = data[0]
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]

    split = [cat(idx[:6], 0), cat(idx[6:8], 0), cat(idx[8:], 0)]

    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask
