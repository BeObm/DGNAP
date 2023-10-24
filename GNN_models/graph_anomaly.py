import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
from search_algo.utils import *


class GNN_Model(nn.Module):
    def __init__(self, model_config):
        super(GNN_Model, self).__init__()

        in_channels = model_config["in_channels"]
        num_class = model_config["num_class"]
        hid_channels = model_config["hidden_channels"]
        self.num_filters = model_config["num_graph_filters"]
        self.K = model_config["num_signals"]
        self.aggr = model_config["aggregation"]
        self.act_fn = model_config["activation"]
        self.attn_fn = model_config["attention"]
        self.normalization = model_config["normalization"]
        self.dropout = model_config["dropout"]
        self.graph_filter = model_config["graph_filter"]

        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels, hid_channels),
                                                 )

        self.filters = nn.ModuleList([self.graph_filter( in_channels=hid_channels,
                                                         out_channels=hid_channels,
                                                         K=self.K,
                                                         aggr=self.aggr,
                                                        normalization=self.normalization,
                                                         bias=False) for _ in range(self.num_filters)])
        self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hid_channels, num_class))

        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, data, label=None):
        set_seed()
        """
        :param label:
        :param x:
        :param edge_index:
        :return:
        """
        x = data.x
        edge_index = data.edge_index
        x = self.linear_transform_in(x)
        h_list = []
        for i, filter_ in enumerate(self.filters):
            h = filter_(x, edge_index)
            h_list.append(h)

        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_filters):
            res += h_filters[:, i, :] * score[:, i]

        y_hat = self.linear_cls_out(res)
        marginal_loss = 0.

        if self.training:
            anomaly_train, normal_train = label
            anomaly_train = anomaly_train.bool()
            normal_train = normal_train.bool()
            normal_bias = score[normal_train][:, 1] - score[normal_train][:, 0]
            anomaly_bias = score[anomaly_train][:, 0] - score[anomaly_train][:, 1]
            normal_bias = torch.clamp(normal_bias, -0.)
            anomaly_bias = torch.clamp(anomaly_bias, -0.)
            normal_bias = torch.mean(normal_bias)
            anomaly_bias = torch.mean(anomaly_bias)
            bias = anomaly_bias + normal_bias
            marginal_loss = bias

        if self.training:
            return y_hat, marginal_loss
        else:
            return y_hat

    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
            (test_attn_anomaly, test_attn_normal)

    @torch.no_grad()
    def evaluating(self, x, y, edge_index, test_index):
        self.eval()
        y_pred = self.forward(x, edge_index)
        y_pred = F.softmax(y_pred, dim=1)[:, 1]
        self.train()
        y_test = y[test_index]
        y_pred = y_pred[test_index]
        y_test = y_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        auc_roc, auc_pr = aucPerformance(y_test, y_pred)
        return auc_roc, auc_pr


def train_function(data, model, criterion, optimizer, beta=.5):
    set_seed()
    anomaly = (data.y == 1)
    normal = (data.y == 0)
    
    idx_train = data.train_mask.bool().to(device)
    model.train()
    optimizer.zero_grad()
    output, bias_loss = model(data, label=(idx_train & anomaly, idx_train & normal))
    loss_train = criterion(output[idx_train], data.y[idx_train]) + bias_loss * beta
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


@torch.no_grad()
def test_function(model, data, typ="val"):
    model.eval()
    data = data.to(device)
    out = model.forward(data)
    # pred = F.softmax(out, dim=1)[:, 1]
    pred = out.argmax(dim=1)
    if typ == 'val':
        mask = data.val_mask.bool()
    if typ == 'test':
        mask = data.test_mask.bool()

    performance_scores = evaluate_model(data.y[mask], pred[mask],typ)
    return performance_scores
