import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(2 * input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(2 * hidden_dim, output_dim, bias=False)

    def forward(self, X, adj):
        null_root = torch.zeros((X.shape[0], 1, X.shape[2])).to(X.device)
        X = torch.cat((null_root, X), dim=1)
        batch_size, node_num, _ = X.shape

        agg = torch.bmm(adj, X)
        X = torch.cat((X, agg), dim=-1)
        X = self.w1(X.view(-1, 2 * self.input_dim)).view(batch_size, node_num, self.hidden_dim)
        X = F.relu(X)

        agg = torch.bmm(adj, X)
        X = torch.cat((X, agg), dim=-1)
        X = self.w2(X.view(-1, 2 * self.hidden_dim)).view(batch_size, node_num, self.output_dim)

        return X[:, 1:, :]
