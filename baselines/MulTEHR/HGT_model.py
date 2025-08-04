import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mymodel.module import SwitchTransformerEncoder, generate_cross_modal_mask, PatchEmbed
from transformers import AutoModel, AutoTokenizer
import os
import random
import numpy as np
from MulTEHR.HGT import ModalityFusionGNN


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache_dir = "pretrained/biobert-base-cased-v1.2"

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, P, Q):
        p = F.softmax(P, dim=-1)
        kl = torch.sum(p * (F.log_softmax(P, dim=-1) - F.log_softmax(Q, dim=-1)))

        return torch.mean(kl)

def cal_unif_loss(feat):
    loss_fcn = KLDivergence()
    feat_min = feat.min(dim=-1, keepdim=True)[0]
    feat_max = feat.max(dim=-1, keepdim=True)[0]
    feat_norm = (feat - feat_min) / (feat_max - feat_min + 1e-6)

    unif_feat = torch.rand_like(feat)

    # Symmetric KL divergence: KL(P‖Q) + KL(Q‖P)
    loss = (loss_fcn(feat_norm, unif_feat) + loss_fcn(unif_feat, feat_norm)) / 2
    return loss

class HGT(nn.Module):
    def __init__(self, ehr_dim=76, num_classes=1, hidden_dim=128, batch_first=True, dropout=0.0, causal = True, device=torch.device('cpu')):
        super(HGT, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.task_embedding = nn.Embedding(40, hidden_dim)
        self.causal = causal

        # Process time series data
        self.ehr_projection = nn.Linear(ehr_dim, hidden_dim)

        # Process image data
        #self.patch_projection = PatchEmbed(patch_size=16, embed_dim=hidden_dim)



        # Process text data
        self.note_projection = AutoModel.from_pretrained(cache_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        self.note_fc = nn.Linear(768, hidden_dim)


        # Process demographic data
        self.demo_projection_race = nn.Linear(5, hidden_dim)  # (race,marital_status,insurance,gender,age)
        self.demo_projection_ms = nn.Linear(5, hidden_dim)
        self.demo_projection_insu = nn.Linear(3, hidden_dim)
        self.demo_projection_gender = nn.Linear(3, hidden_dim)
        self.demo_projection_age = nn.Linear(1, hidden_dim)

        
        # fusion_moe
        

        self.modal_gnn = ModalityFusionGNN(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, n_layers=2, dropout=0.1, causal = False)

        self.dense_layer_mortality = nn.Linear(hidden_dim, 1)
        self.dense_layer_decomp = nn.Linear(hidden_dim, 1)
        self.dense_layer_ph = nn.Linear(hidden_dim, 25)
        self.dense_layer_los = nn.Linear(hidden_dim, 10)
        self.dense_layer_readm = nn.Linear(hidden_dim, 1)
        



    def forward(self, ehr, ehr_lengths, use_ehr,  demo, use_demo, task_index, labels, criterion):
        task_embed = self.task_embedding(task_index).unsqueeze(1)
        
        # Time series
        ehr_embed = self.ehr_projection(ehr)
        
        cxr
        cxr_embed = self.patch_projection(img)
        


        Text
        with torch.no_grad():
            encoding = self.tokenizer(note, padding=True, truncation=True, max_length=512, add_special_tokens=False, return_tensors='pt')
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            # if there is no text in this batch
            if attention_mask.sum()!=0:
                outputs = self.note_projection(input_ids, attention_mask=attention_mask)
                note_embed = outputs.last_hidden_state
            else:
                note_embed = torch.zeros((len(note), 1, self.note_fc.in_features)).to(self.device)
                attention_mask = torch.zeros((len(note), 1)).int().to(self.device)

        note_embed = self.note_fc(note_embed)
        
        # Demographics
        race = np.array([col[0] for col in demo])
        ms = np.array([col[1] for col in demo])
        insu = np.array([col[2] for col in demo])
        gender = np.array([col[3] for col in demo])
        #age = np.array([col[4] for col in demo])
        race_embed = self.demo_projection_race(torch.from_numpy(race).float().to(self.device))
        ms_embed = self.demo_projection_ms(torch.from_numpy(ms).float().to(self.device))
        insu_embed = self.demo_projection_insu(torch.from_numpy(insu).float().to(self.device))
        gender_embed = self.demo_projection_gender(torch.from_numpy(gender).float().to(self.device))
        #age_embed = self.demo_projection_age(torch.from_numpy(age).float().to(self.device))
        demo_embed = torch.stack([race_embed, ms_embed, insu_embed, gender_embed], dim = 1)

        multimodal_embed = torch.cat((ehr_embed, demo_embed), dim=1)


        ori_output, final_mm_embed = self.modal_gnn(multimodal_embed)
        
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
        elif task_index[0] == 5:
            out = self.dense_layer_drg(final_mm_embed)
            scores = out
        else:
            out = self.dense_layer_ph(final_mm_embed)
            scores = torch.sigmoid(out)

        pred_loss = criterion(scores,labels)
        unif_loss = cal_unif_loss(final_mm_embed) if self.causal else 0
        
        loss = pred_loss + unif_loss * 0.00001
        

        # print("fusion repre:", fusion_embed.shape)
        # print("proportion:",modality_weights)
        # print("loss:", fusion_moe_loss)
        # print(saliencies)

        if self.training is True:
            return scores, loss
            
        else:
            return scores
        
