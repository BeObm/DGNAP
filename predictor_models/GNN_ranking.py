# -*- coding: utf-8 -*-

# from sklearn.preprocessing import OneHotEncoder


from search_algo.utils import *

from torch_geometric.nn import MessagePassing
from search_space_manager.search_space import *
from search_space_manager.sample_models import *
from predictor_models.utils import *
from torch_geometric.nn import global_add_pool  , global_mean_pool, global_max_pool
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn.norm import GraphNorm, InstanceNorm, BatchNorm
import torch.nn.functional as F

from settings.config_file import *


class Predictor(MessagePassing):
    def __init__(self, in_channels, dim, out_channels, drop_out):
        super(Predictor, self).__init__()
        #         self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels)
        # print("in channels dim =",in_channels)
        self.conv1 = GraphConv(in_channels, dim, aggr="add")

        self.conv2 = GraphConv(dim, dim, aggr="add")

        self.drop_out = drop_out
        # self.normalize = InstanceNorm(dim)
        self.graphnorm = GraphNorm(dim)
        self.linear = Linear(dim, 64)
        self.linear2 = Linear(64, out_channels)

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)

       
        x = global_mean_pool(x, batch)
        # x = self.graphnorm(x)

        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x


def train_predictor(model, dataloader, criterion, optimizer,accelerator):
    model.train()
    total_loss = 0

    for data in dataloader:
        targets = data.y
        optimizer.zero_grad()
        output = model(data)
        target = get_target(targets,output).to(accelerator.device)

        loss = criterion(output, targets,target)
        accelerator.backward(loss)
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return loss.item()


@torch.no_grad()
def test_predictor(accelerator,model, test_loader, metrics_list, title):
    model.eval()
    ped_list, label_list = [], []
    for data in test_loader:
        targets = data.y
        pred = model(data)

        all_targets = accelerator.gather(targets)
        all_pred = accelerator.gather(pred)

        ped_list = np.append(ped_list, all_pred.cpu().detach().numpy())
        label_list = np.append(label_list, all_targets.cpu().detach().numpy())
    predictor_performance = evaluate_model_predictor(label_list, ped_list, metrics_list, title)

    return predictor_performance