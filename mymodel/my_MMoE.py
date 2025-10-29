import collections.abc
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from mymodel.module import MLP, SparseDispatcher



class MMoE(nn.Module):
    def __init__(self, query_size, input_size, output_size, num_experts, num_tasks, hidden_size, noisy_gating=False, k=4, device=torch.device('cpu')):
        super(MMoE, self).__init__()
        self.query_size = query_size
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.noisy_gating = noisy_gating
        self.k = k
        self.device = device

        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        self.gates = nn.ModuleList([nn.Linear(self.input_size, self.num_experts) for _ in range(self.num_tasks)])
        self.w_noise = nn.ModuleList([nn.Linear(self.input_size, self.num_experts)for _ in range(self.num_tasks)])


        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

        # Specific classifier
        self.dense_layer_mortality = nn.Linear(hidden_size, 1)
        self.dense_layer_decomp = nn.Linear(hidden_size, 1)
        self.dense_layer_ph = nn.Linear(hidden_size, 25)
        self.dense_layer_los = nn.Linear(hidden_size, 10)
        self.dense_layer_readm = nn.Linear(hidden_size, 1)

        self.mm_layernorm = nn.LayerNorm(hidden_size)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, task_idx, x, train, noise_epsilon=1e-2):
        gate_logits = self.gates[task_idx](x)
        clean_logits = gate_logits#.unsqueeze(0).repeat(x.shape[0],1)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise[task_idx](x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

  
    def forward(self, mm_embed, task_index, true_y, criterion, loss_coef=1e-2):
        # create task-to-module graph A
        outputs = []
        mmoe_total_loss = 0.0
        all_expert_outputs = None
        x = mm_embed.reshape(-1, mm_embed.shape[2])

        for task_idx in range(self.num_tasks):

            gate_weights, load = self.noisy_top_k_gating(task_idx ,x, self.training)
            importance = gate_weights.sum(0)
            task_loss = self.cv_squared(importance) + self.cv_squared(load)
            mmoe_total_loss += task_loss

            dispatcher = SparseDispatcher(self.num_experts, gate_weights)
            expert_inputs = dispatcher.dispatch(x)
            gate_weights = dispatcher.expert_to_gates()
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

            y = dispatcher.combine(expert_outputs)
            outputs.append(y)

        mmoe_total_loss *= loss_coef

        output = torch.stack(outputs)
        tmp_moe = output[task_index[0]]
        mm_moe = tmp_moe.reshape(mm_embed.shape[0], mm_embed.shape[1], mm_embed.shape[2]).sum(dim=1)
        final_mm_embed = self.mm_layernorm(mm_moe)


        if task_index[0] == 0:
            out = self.dense_layer_mortality(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 1:
            out = self.dense_layer_decomp(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 3:
            out = self.dense_layer_los(final_mm_embed)
            scores = out
        elif task_index[0] == 4:
            out = self.dense_layer_readm(final_mm_embed)
            scores = torch.sigmoid(out)
        else:
            out = self.dense_layer_ph(final_mm_embed)
            scores = torch.sigmoid(out)

        # Calculate Prediction task loss
        pred_loss = criterion(scores, true_y)

        return scores, mmoe_total_loss, pred_loss


class MoE(nn.Module):
    def __init__(self, query_size, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4,device=torch.device('cpu')):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.k = k
        self.device = device
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.task_gate = nn.Parameter(torch.zeros(query_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

        # Specific classifier
        self.dense_layer_mortality = nn.Linear(hidden_size, 1)
        self.dense_layer_decomp = nn.Linear(hidden_size, 1)
        self.dense_layer_ph = nn.Linear(hidden_size, 25)
        self.dense_layer_los = nn.Linear(hidden_size, 10)
        self.dense_layer_readm = nn.Linear(hidden_size, 1)

        self.mm_layernorm = nn.LayerNorm(hidden_size)


    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        modal_logits = x @ self.w_gate

        clean_logits = modal_logits
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, mm_embed, task_index, true_y, criterion, loss_coef=1e-2):
        x = mm_embed.reshape(-1, mm_embed.shape[2])
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        tmp_moe = y
        mm_moe = tmp_moe.reshape(mm_embed.shape[0], mm_embed.shape[1], mm_embed.shape[2]).sum(dim=1)
        final_mm_embed = self.mm_layernorm(mm_moe)

        if task_index[0] == 0:
            out = self.dense_layer_mortality(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 1:
            out = self.dense_layer_decomp(final_mm_embed)
            scores = torch.sigmoid(out)
        elif task_index[0] == 3:
            out = self.dense_layer_los(final_mm_embed)
            scores = out
        elif task_index[0] == 4:
            out = self.dense_layer_readm(final_mm_embed)
            scores = torch.sigmoid(out)
        else:
            out = self.dense_layer_ph(final_mm_embed)
            scores = torch.sigmoid(out)

        pred_loss = criterion(scores, true_y)

        # Calculate Prediction task loss
        pred_loss = criterion(scores, true_y)

        return scores, loss, pred_loss
