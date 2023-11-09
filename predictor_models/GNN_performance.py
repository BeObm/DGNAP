# -*- coding: utf-8 -*-

# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
from tqdm import tqdm
import torch.nn as nn
import math
import scipy.stats as stats
from search_algo.utils import *
from search_space_manager.map_functions import map_activation
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from search_space_manager.search_space import *
from search_space_manager.sample_models import *
from predictor_models.utils import *
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.nn import global_add_pool  # global_mean_pool, global_max_pool,
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import glob
import pandas as pd
from torch_geometric.nn.norm import GraphNorm
from sklearn.linear_model import SGDRegressor, LassoCV
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, SAGEConv, GATConv, LEConv, GENConv, GeneralConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm, InstanceNorm, BatchNorm

from settings.config_file import *


class Predictor(MessagePassing):
    def __init__(self, in_channels, dim, out_channels, drop_out):
        super(Predictor, self).__init__()
        #         self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels)
        # print("in channels dim =",in_channels)
        self.conv1 = GraphSAGE(in_channels,128,4, dim,)

        self.conv2 = GraphSAGE(dim,128,3, dim)
        self.drop_out = drop_out
        # self.normalize = InstanceNorm(dim)
        self.graphnorm = GraphNorm(dim)
        self.linear = Linear(dim, 64)
        self.linear2 = Linear(64, out_channels)

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = global_add_pool(x, batch)
        # x = self.graphnorm(x)

        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x


def train_predictor(predictor_model, train_loader, criterion, optimizer):
    predictor_model.train()

    total_loss = 0
    for data in train_loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)
        optimizer.zero_grad()
        output = predictor_model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)

        loss.backward()

        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return loss.item()


@torch.no_grad()
def test_predictor(model, test_loader, metrics_list, title):
    model.eval()
    ped_list, label_list = [], []
    for data in test_loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        pred = model(data.x, data.edge_index, data.batch)
        ped_list = np.append(ped_list, pred.cpu().detach().numpy())
        label_list = np.append(label_list, data.y.cpu().detach().numpy())
    predictor_performance = evaluate_model_predictor(label_list, ped_list, metrics_list, title)

    return predictor_performance
