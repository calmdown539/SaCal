import collections.abc
from itertools import repeat
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)          # (H, W)
        patch_size = to_2tuple(patch_size)      # (P, P)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])       # N = (H // P) * (W // P)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



def generate_cross_modal_mask(ehr_cls_index=None, cxr_cls_index=None, note_cls_index=None, demo_cls_index=None, total_lens=None):    
    
    mask = torch.ones(total_lens, total_lens)
    
    # # task
    # mask[0, :] = 0
    if cxr_cls_index == None and note_cls_index==None:
        mask[1, 1] = 0
        mask[1, ehr_cls_index + 1: demo_cls_index] = 0
        mask[1, demo_cls_index + 1: total_lens] = 0

        mask[ehr_cls_index, ehr_cls_index: demo_cls_index] = 0
        mask[ehr_cls_index+1: demo_cls_index, ehr_cls_index+1: demo_cls_index] = 0

        mask[demo_cls_index, demo_cls_index: total_lens] = 0
        mask[demo_cls_index+1: total_lens, demo_cls_index+1: total_lens] = 0


    ##############ehr,note,demo##############
    elif cxr_cls_index == None:
        # m1+m2+m3
        mask[1, 1] = 0
        mask[1, ehr_cls_index + 1: note_cls_index] = 0
        mask[1, note_cls_index + 1: demo_cls_index] = 0
        mask[1, demo_cls_index + 1: total_lens] = 0
        
        # m1+m2 mask
        mask[2, 2] = 0
        mask[2, ehr_cls_index + 1: note_cls_index] = 0
        mask[2, note_cls_index + 1: demo_cls_index] = 0

        # m1+m3 mask
        mask[3, 3] = 0
        mask[3, ehr_cls_index + 1: note_cls_index] = 0
        mask[3, demo_cls_index + 1: total_lens] = 0

        # m2+m3 mask
        mask[4, 4] = 0
        mask[4, note_cls_index + 1: demo_cls_index] = 0
        mask[4, demo_cls_index + 1: total_lens] = 0

        # m1
        mask[ehr_cls_index, ehr_cls_index: note_cls_index] = 0
        mask[ehr_cls_index+1: note_cls_index, ehr_cls_index+1: note_cls_index] = 0

        # m2
        mask[note_cls_index, note_cls_index: demo_cls_index] = 0
        mask[note_cls_index+1: demo_cls_index, note_cls_index+1: demo_cls_index] = 0

        # m3
        mask[demo_cls_index, demo_cls_index: total_lens] = 0
        mask[demo_cls_index+1: total_lens, demo_cls_index+1: total_lens] = 0
    
    elif demo_cls_index == None:
        # m1+m2+m3
        mask[1, 1] = 0
        mask[1, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[1, cxr_cls_index + 1: note_cls_index] = 0
        mask[1, note_cls_index + 1: total_lens] = 0

        # m1+m2 mask
        mask[2, 2] = 0
        mask[2, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[2, cxr_cls_index + 1: note_cls_index] = 0

        # m1+m3 mask
        mask[3, 3] = 0
        mask[3, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[3, note_cls_index + 1: total_lens] = 0

        # m2+m3 mask
        mask[4, 4] = 0
        mask[4, cxr_cls_index + 1: note_cls_index] = 0
        mask[4, note_cls_index + 1: total_lens] = 0

        # m1
        mask[ehr_cls_index, ehr_cls_index: cxr_cls_index] = 0
        mask[ehr_cls_index+1: cxr_cls_index, ehr_cls_index+1: cxr_cls_index] = 0

        # m2
        mask[cxr_cls_index, cxr_cls_index: note_cls_index] = 0
        mask[cxr_cls_index+1: note_cls_index, cxr_cls_index+1: note_cls_index] = 0

        # m3
        mask[note_cls_index, note_cls_index: total_lens] = 0
        mask[note_cls_index+1: total_lens, note_cls_index+1: total_lens] = 0
    else:
        ##############4 modals##############
        # m1+m2+m3+m4
        mask[0, 0] = 0
        mask[0, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[0, cxr_cls_index + 1: note_cls_index] = 0
        mask[0, note_cls_index + 1: demo_cls_index] = 0
        mask[0, demo_cls_index + 1: total_lens] = 0
        
        # m1+m2 mask
        mask[1, 1] = 0
        mask[1, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[1, cxr_cls_index + 1: note_cls_index] = 0

        # m1+m3 mask
        mask[2, 2] = 0
        mask[2, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[2, note_cls_index + 1: demo_cls_index] = 0
        
        # m1+m4 mask
        mask[3, 3] = 0
        mask[3, ehr_cls_index + 1: cxr_cls_index] = 0
        mask[3, demo_cls_index + 1: total_lens] = 0

        # m2+m3 mask
        mask[4, 4] = 0
        mask[4, cxr_cls_index + 1: note_cls_index] = 0
        mask[4, note_cls_index + 1: demo_cls_index] = 0

        # m2+m4 mask
        mask[5, 5] = 0
        mask[5, cxr_cls_index + 1: note_cls_index] = 0
        mask[5, demo_cls_index + 1: total_lens] = 0

        # m3+m4 mask
        mask[6, 6] = 0
        mask[6, note_cls_index + 1: demo_cls_index] = 0
        mask[6, demo_cls_index + 1: total_lens] = 0

        # m1
        mask[ehr_cls_index, ehr_cls_index: cxr_cls_index] = 0
        mask[ehr_cls_index+1: cxr_cls_index, ehr_cls_index+1: cxr_cls_index] = 0

        # m2
        mask[cxr_cls_index, cxr_cls_index: note_cls_index] = 0
        mask[cxr_cls_index+1: note_cls_index, cxr_cls_index+1: note_cls_index] = 0

        # m3
        mask[note_cls_index, note_cls_index: demo_cls_index] = 0
        mask[note_cls_index+1: demo_cls_index, note_cls_index+1: demo_cls_index] = 0

        # m4
        mask[demo_cls_index, demo_cls_index: total_lens] = 0
        mask[demo_cls_index+1: total_lens, demo_cls_index+1: total_lens] = 0
    

    
    return mask



class CoSFuserMoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, capacity_factor=1.0, alpha = 0.001):
        super(CoSFuserMoELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.alpha = alpha

        self.router = nn.Linear(input_dim, num_experts)
        self.register_buffer("latest_saliency", torch.zeros(num_experts))
        self.experts = nn.ModuleList([
            MLP(input_dim, input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.saved_outputs = [[] for _ in range(num_experts)]
        self.saved_grads = [[] for _ in range(num_experts)]
    
    def compute_load_balancing_loss(self, router_probs, expert_mask):

        mean_prob = router_probs.mean(dim=0)  # [num_experts]
        expert_usage = expert_mask.sum(dim=0) / expert_mask.sum()  # [num_experts]
        loss = self.num_experts * torch.sum(mean_prob * expert_usage)
        return loss

    def get_expert_saliency(self,batch_size,input_dim):
        saliencies = []
        for i in range(self.num_experts):
            if len(self.saved_outputs[i]) > 0 and len(self.saved_grads[i]) > 0:
                try:
                    outputs = torch.cat(self.saved_outputs[i], dim=0)
                    outputs = outputs.view(batch_size,-1,input_dim).sum(dim=1)

                    grads = torch.cat(self.saved_grads[i], dim=0)

                    grads = grads.view(batch_size,-1,input_dim).sum(dim=1)

                    saliency = (outputs.abs() * grads.abs()).sum(dim=1).mean()
                    #print('sal:',saliency)
                    saliencies.append(saliency.item())
                    
                except Exception as e:
                    print(f"Error computing saliency for expert {i}: {e}")
                    saliencies.append(0.0)
            else:
                saliencies.append(0.0)
        return torch.tensor(saliencies)
    def clear_saved_states(self):
        for i in range(self.num_experts):
            self.saved_outputs[i].clear()
            self.saved_grads[i].clear()
    
    def forward(self, x):

        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        if self.training:
            # Normalize saliency
            sal = self.latest_saliency.detach()
            #print('1.',sal)
            sal = sal / (sal.sum() + 1e-8)  # [num_experts]
            #print(sal)
            router_logits = router_logits + self.alpha * sal 
        router_probs = F.softmax(router_logits, dim=-1)
        
        max_prob, max_indices = router_probs.max(dim=-1)  # [batch_size * seq_len]
        expert_mask = F.one_hot(max_indices, num_classes=self.num_experts).float()
        balance_loss = self.compute_load_balancing_loss(router_probs, expert_mask)
        
        output = torch.zeros_like(x_flat)  # [batch_size * seq_len, input_dim]
        #print("expert mask:",expert_mask)
        
        for expert_idx in range(self.num_experts):
            token_mask = (max_indices == expert_idx)  # [batch_size * seq_len]
            if token_mask.sum() == 0:
                continue 
            
            selected_tokens = x_flat[token_mask]  # [num_selected, input_dim]
            
            expert_output = self.experts[expert_idx](selected_tokens)  # [num_selected, input_dim]
            output[token_mask] = expert_output
            sparse_output = torch.zeros_like(output)  
            sparse_output[token_mask] = expert_output
        
            if self.training is True:
                self.saved_outputs[expert_idx].append(sparse_output)
                def save_grad_hook(grad, idx=expert_idx):
                    self.saved_grads[idx].append(grad)
                output.register_hook(save_grad_hook)
                        
        output = output * max_prob.unsqueeze(-1)  # [batch_size * seq_len, input_dim]
        
        output = output.view(batch_size, seq_len, input_dim)  # [batch_size, seq_len, input_dim]
        if self.training is True:
            layer_saliency = self.get_expert_saliency(batch_size,input_dim)
            self.latest_saliency.copy_(layer_saliency)
            max_sal_idx = torch.argmax(layer_saliency).item()
            #print(f"Expert with highest sal:{max_sal_idx}")

            
            selected_token_indices = (max_indices == max_sal_idx).nonzero(as_tuple=False).squeeze(-1)
            batch_positions = selected_token_indices // seq_len  # shape: [N]
            seq_positions = selected_token_indices % seq_len     # shape: [N]
            token_positions = list(zip(batch_positions.tolist(), seq_positions.tolist()))

            #print(f"Tokens processed by Expert {max_sal_idx}: {token_positions}")
        else:
            layer_saliency = None
            seq_positions = None

        
        #print('layer_sal',layer_saliency)
        self.clear_saved_states()

        return output, balance_loss, layer_saliency, seq_positions

class CoSFuserLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead=2, num_experts=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.switch_ffn = CoSFuserMoELayer(d_model, dim_feedforward, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, cross_mask=None, pad_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask = cross_mask, key_padding_mask = pad_mask)
        x = self.norm1(x + attn_output)
        ffn_output, balance_loss, layer_saliency,token_positions  = self.switch_ffn(x)
        output = self.norm2(x + ffn_output)
        return output, balance_loss, layer_saliency, token_positions

class CoSFuser(nn.Module):
    def __init__(self, d_model,  dim_feedforward, nhead = 2, num_layers=2, num_experts = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            CoSFuserLayer(d_model, dim_feedforward, nhead, num_experts)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None, src_key_padding_mask=None, loss_coef = 0.0):
        total_loss = 0
        saliencies = []
        for layer in self.layers:
            src, loss, layer_saliency, token_positions = layer(src, mask, src_key_padding_mask)
            total_loss += loss
            saliencies.append(layer_saliency)
        total_loss *= loss_coef
        if self.training is True:
            return src, total_loss, torch.stack(saliencies), token_positions
        else:
            return src, total_loss, None, None
