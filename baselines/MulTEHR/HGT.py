
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedGNNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(BatchedGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, node_features, adj_matrix):


        transformed_features = self.linear(node_features)


        # [B, N, N] @ [B, N, C_out] -> [B, N, C_out]
        aggregated_features = torch.bmm(adj_matrix, transformed_features)

        new_node_features = self.activation(aggregated_features)
        
        return new_node_features

class ModalityFusionGNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1, causal = False):

        super(ModalityFusionGNN, self).__init__()
        

        self.adapt_layer = nn.Linear(input_dim, hidden_dim)
        self.causal = causal

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(BatchedGNNLayer(hidden_dim, hidden_dim))
        if causal:
            self.rand_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.rand_layers.append(BatchedGNNLayer(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        batch_size, num_modalities, _ = x.shape
        device = x.device


        h = F.gelu(self.adapt_layer(x))
        h = self.dropout(h)


        adj = torch.ones(batch_size, num_modalities, num_modalities, device=device)

        adj = F.normalize(adj, p=1, dim=2)
        
        rand = h

        for layer in self.layers:
            h = layer(h, adj)

        output = h

        if self.causal:
            for rand_layer in self.rand_layers:
                rand = rand_layer(rand, adj)
            interv = rand + h
            output = interv

        pooled = torch.mean(output, dim=1)
        
        
        return output, pooled
