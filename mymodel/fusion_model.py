import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import length_to_mask
from transformers import AutoModel, AutoTokenizer
import os
import random
import numpy as np
from mymodel.module import CoSFuser, generate_cross_modal_mask, PatchEmbed
from mymodel.graph_MMoE import GraphMMoE


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_dir = "mymodel/pretrained/biobert-base-cased-v1.2"

import torch

### count tokens for interpretation
def count_seq_positions_in_bins(seq_positions, bins):
    seq_positions = torch.tensor(seq_positions) if not isinstance(seq_positions, torch.Tensor) else seq_positions
    bin_counts = []

    for start, end in bins:
        count = ((seq_positions >= start) & (seq_positions < end)).sum().item()
        bin_counts.append(count)

    return bin_counts

### decor reg calculation
def correlation_matrix(Z1, Z2):
    Z1_centered = Z1 - Z1.mean(dim=0, keepdim=True)
    Z2_centered = Z2 - Z2.mean(dim=0, keepdim=True)

    cov = (Z1_centered * Z2_centered).mean(dim=0)  
    std1 = Z1_centered.std(dim=0)
    std2 = Z2_centered.std(dim=0)

    corr = cov / (std1 * std2 + 1e-8)  
    return corr

def decorrelation_loss(phi_outputs, lambda_decor=0.005):
    k = 4
    phi_outputs = phi_outputs.view(phi_outputs.shape[0],4,128)
    loss = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            corr = correlation_matrix(phi_outputs[:,i,:].squeeze(), phi_outputs[:,j,:].squeeze())
            frob_norm_sq = torch.norm(corr, p='fro') ** 2
            loss += frob_norm_sq
    return lambda_decor * loss

def calculate_ortho_loss(input_vec, lambda_fd = 0.5):
    x = input_vec - torch.mean(input_vec, axis=2).unsqueeze(2).repeat(1, 1, input_vec.shape[2])
    cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
    loss = (torch.sum(cov_matrix ** 2) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2))/(cov_matrix.shape[0]*(cov_matrix.shape[1]-1)*(cov_matrix.shape[2]-1))
    return loss * lambda_fd 


class OurModel(nn.Module):
    def __init__(self, ehr_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, layers=3, expert_k=2, expert_total=5, device=torch.device('cpu')):
        super(OurModel, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        

        # Process time series data
        self.ehr_projection = nn.Linear(ehr_dim, hidden_dim)
        self.ehr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ehr_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Process image data
        self.patch_projection = PatchEmbed(patch_size=16, embed_dim=hidden_dim)
        num_patches = (224 // 16) * (224 // 16)
        self.cxr_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.cxr_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Process text data
        self.note_projection = AutoModel.from_pretrained(cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        self.note_fc = nn.Linear(768, hidden_dim)
        self.note_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.note_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        # Process demographic data
        self.demo_projection_race = nn.Linear(5, hidden_dim)  
        self.demo_projection_ms = nn.Linear(5, hidden_dim)
        self.demo_projection_insu = nn.Linear(3, hidden_dim)
        self.demo_projection_gender = nn.Linear(3, hidden_dim)
        self.demo_projection_age = nn.Linear(1, hidden_dim)

        self.demo_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.demo_pos_embed = nn.Parameter(torch.zeros(1, 600, hidden_dim))

        self.cross_cls_tokens = nn.Parameter(torch.zeros(6, 1, hidden_dim))
        self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # fusion_moe
        self.transformer_fusion = CoSFuser(d_model = hidden_dim, dim_feedforward=hidden_dim*4, nhead=2, num_layers=layers, num_experts=4)

        # MMoe
        self.graphmmoe = GraphMMoE(hidden_dim, hidden_dim, hidden_dim, expert_total, 5, hidden_dim, noisy_gating=False, k=expert_k, device=device)



    def forward(self, ehr, ehr_lengths, use_ehr, img, use_img, note, use_note, demo, use_demo, task_index, labels, criterion):
        ### for decor reg
        single_modal = torch.FloatTensor().to(self.device)
        # Time series
        ehr_embed = self.ehr_projection(ehr)
        single_modal = torch.cat((single_modal,torch.mean(ehr_embed,dim=1)),dim=1)
        ehr_cls_tokens = self.ehr_cls_token.repeat(ehr_embed.shape[0], 1, 1)
        ehr_embed = ehr_embed + self.ehr_pos_embed[:, :ehr_embed.shape[1], :]
        ehr_embed = torch.cat((ehr_cls_tokens, ehr_embed), dim=1)
        ehr_lengths = torch.tensor(ehr_lengths).to(self.device)
        if use_ehr.sum()!=0:
            ehr_pad_mask = length_to_mask(ehr_lengths+use_ehr)
        else:
            ehr_pad_mask = length_to_mask(ehr_lengths+use_ehr, max_len=2)

        #cxr
        cxr_embed = self.patch_projection(img)
        single_modal = torch.cat((single_modal,torch.mean(cxr_embed,dim=1)),dim=1)
        cxr_cls_tokens = self.cxr_cls_token.repeat(cxr_embed.shape[0], 1, 1)
        cxr_embed = cxr_embed + self.cxr_pos_embed[:, :cxr_embed.shape[1], :]
        cxr_embed = torch.cat((cxr_cls_tokens, cxr_embed), dim=1)
        cxr_pad_mask = length_to_mask(use_img, max_len=1).repeat(1, cxr_embed.shape[1])


        # Text
        with torch.no_grad():
            encoding = self.tokenizer(note, padding=True, truncation=True, max_length=512, add_special_tokens=False, return_tensors='pt')
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            if attention_mask.sum()!=0:
                outputs = self.note_projection(input_ids, attention_mask=attention_mask)
                note_embed = outputs.last_hidden_state
            else:
                note_embed = torch.zeros((len(note), 1, self.note_fc.in_features)).to(self.device)
                attention_mask = torch.zeros((len(note), 1)).int().to(self.device)

        note_embed = self.note_fc(note_embed)
        single_modal = torch.cat((single_modal,torch.mean(note_embed,dim=1)),dim=1)
        note_cls_tokens = self.note_cls_token.repeat(note_embed.shape[0], 1, 1)
        note_embed = note_embed + self.note_pos_embed[:, :note_embed.shape[1], :]
        if attention_mask.sum()!=0:
            note_embed = torch.cat((note_cls_tokens, note_embed), dim=1)
        else:
            note_embed = note_cls_tokens

        if attention_mask.sum()!=0:
            note_pad_mask = length_to_mask(attention_mask.sum(dim=1)+use_note)
        else:
            note_pad_mask = length_to_mask(attention_mask.sum(dim=1)+use_note, max_len=1)

        # Demographics
        race = np.array([col[0] for col in demo])
        ms = np.array([col[1] for col in demo])
        insu = np.array([col[2] for col in demo])
        gender = np.array([col[3] for col in demo])
        age = np.array([col[4] for col in demo])
        race_embed = self.demo_projection_race(torch.from_numpy(race).float().to(self.device))
        ms_embed = self.demo_projection_ms(torch.from_numpy(ms).float().to(self.device))
        insu_embed = self.demo_projection_insu(torch.from_numpy(insu).float().to(self.device))
        gender_embed = self.demo_projection_gender(torch.from_numpy(gender).float().to(self.device))
        age_embed = self.demo_projection_age(torch.from_numpy(age).float().to(self.device))
        demo_embed = torch.stack([race_embed, ms_embed, insu_embed, gender_embed, age_embed], dim = 1)
        single_modal = torch.cat((single_modal,torch.mean(demo_embed,dim=1)),dim=1)

        demo_cls_tokens = self.demo_cls_token.repeat(demo_embed.shape[0], 1, 1)
        demo_embed = demo_embed + self.demo_pos_embed[:, :demo_embed.shape[1], :]
        demo_embed = torch.cat((demo_cls_tokens, demo_embed), dim=1)
        demo_pad_mask = length_to_mask(use_demo,max_len=1).repeat(1,demo_embed.shape[1])

        

        # Multimodal fusion
        multimodal_cls_tokens = self.mm_cls_token
        # 6 types of pair-wise modality combinations from 4 modalities
        for i in range(6):
            multimodal_cls_tokens = torch.cat((multimodal_cls_tokens, self.cross_cls_tokens[i].unsqueeze(0)), dim=1)
        multimodal_cls_tokens = multimodal_cls_tokens.repeat(ehr_embed.shape[0], 1, 1)

        
        multimodal_embed = torch.cat((multimodal_cls_tokens, ehr_embed, cxr_embed, note_embed, demo_embed), dim=1)

        
        cls_pad_mask = length_to_mask(7*torch.ones(ehr_embed.shape[0]).to(self.device), max_len=7) 
    

        multimodal_pad_mask = torch.cat((cls_pad_mask, ehr_pad_mask, cxr_pad_mask, note_pad_mask, demo_pad_mask), dim=1)

        ehr_cls_index = 7
        cxr_cls_index = ehr_cls_index + ehr_embed.shape[1]
        note_cls_index = cxr_cls_index + cxr_embed.shape[1]
        demo_cls_index = note_cls_index + note_embed.shape[1]

    
        ### mask matrix for fusion
        cross_cls_mask = generate_cross_modal_mask(ehr_cls_index, cxr_cls_index, note_cls_index, demo_cls_index, multimodal_embed.shape[1]).to(self.device)
        ### fusion
        fusion_embed, moe_loss,saliencies, token_positions = self.transformer_fusion(multimodal_embed, mask=cross_cls_mask, src_key_padding_mask=multimodal_pad_mask) 
        
        ### fusion results
        cross_mm_embed = fusion_embed[:, 0:ehr_cls_index]
        single_mm_embed = torch.cat((fusion_embed[:, ehr_cls_index].unsqueeze(1), fusion_embed[:, cxr_cls_index].unsqueeze(1), fusion_embed[:, note_cls_index].unsqueeze(1), fusion_embed[:, demo_cls_index].unsqueeze(1)), dim=1)
        mm_embed = torch.cat((cross_mm_embed, single_mm_embed), dim=1)
        
        ### multi-task graph learning
        scores, mmoe_loss, pred_loss, graph_loss = self.graphmmoe( mm_embed, task_index, labels, criterion)
        
        # Calculate needed loss
        # 1. Decorrelation loss of combination representation
        ortho_loss = calculate_ortho_loss(mm_embed)

        # 2. Decorrelation loss of each modality representation
        single_decor_loss = decorrelation_loss(single_modal)

        # 3. Total loss
        loss = pred_loss + ortho_loss + mmoe_loss + graph_loss + moe_loss + single_decor_loss 
        

        if self.training is True:
            #interpretation：
            bins = [(ehr_cls_index,cxr_cls_index),(cxr_cls_index,note_cls_index),(note_cls_index,demo_cls_index),(demo_cls_index,demo_cls_index+5)]
            count_token_in_modal = count_seq_positions_in_bins(token_positions,bins)
            max_idx = count_token_in_modal.index(max(count_token_in_modal))
            if max_idx == 0:
                key_modal = 'ehr'
            elif max_idx == 1:
                key_modal = 'cxr'
            elif max_idx == 2:
                key_modal = 'note'
            else:
                key_modal = 'demo'
            return scores, loss, key_modal
            
        else:
            return scores
        
