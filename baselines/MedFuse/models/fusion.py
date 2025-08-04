from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
from .loss import RankingLoss, CosineLoss, KLDivLoss
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, args, ehr_model, cxr_model, device):
	
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.device = device

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        

        # if self.args.labels_set == 'radiology':
        #     target_classes = self.args.vision_num_classes
        #     lstm_in = self.cxr_model.feats_dim
        #     projection_in = self.ehr_model.feats_dim

        # import pdb; pdb.set_trace()
        self.projection = nn.Linear(projection_in, lstm_in)
        # feats_dim = 2 * self.ehr_model.feats_dim
        # feats_dim = self.ehr_model.feats_dim + self.cxr_model.feats_dim

        # self.fused_cls = nn.Sequential(
        #     nn.Linear(feats_dim, self.args.num_classes),
        #     nn.Sigmoid()
        # )

        # self.align_loss = CosineLoss()
        # self.kl_loss = KLDivLoss()

        hidden_dim = self.args.dim
        cache_dir = '/data2/linfx/FlexCare-main/mymodel/pretrained/biobert-base-cased-v1.2'
        # Process demographic data
        self.demo_projection_race = nn.Linear(5, hidden_dim)  # (race,marital_status,insurance,gender,age)
        self.demo_projection_ms = nn.Linear(5, hidden_dim)
        self.demo_projection_insu = nn.Linear(3, hidden_dim)
        self.demo_projection_gender = nn.Linear(3, hidden_dim)
        self.demo_projection_age = nn.Linear(1, hidden_dim)
        # Process note data
        self.note_projection = AutoModel.from_pretrained(cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        self.note_fc = nn.Linear(768, hidden_dim)
        

        self.lstm_fused_cls =  nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        ) 

        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout = 0.0)


    def process_demo_data(self,demo):
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
        return demo_embed
    def process_note_data(self,note):
        # Text
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
            
        return note_embed
    
    def forward(self, x, img=None, note=None, demo=None, seq_lengths=None, pairs=None):

        demo_feats = self.process_demo_data(demo)
        #note_feats = self.process_note_data(note)

        if self.args.labels_set == 'radiology':
            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            _, _ , cxr_feats = self.cxr_model(img)

            feats = cxr_feats[:,None,:]

            ehr_feats = self.projection(ehr_feats)

            ehr_feats[list(~np.array(pairs))] = 0
            feats = torch.cat([feats, ehr_feats[:,None,:]], dim=1)
        else:

            _ , ehr_feats = self.ehr_model(x, seq_lengths)
            
            #cxr_feats = self.cxr_model(img)
            #cxr_feats = self.projection(cxr_feats)

            #cxr_feats[list(~np.array(pairs))] = 0
            if len(ehr_feats.shape) == 1:
                # print(ehr_feats.shape, cxr_feats.shape)
                # import pdb; pdb.set_trace()
                feats = ehr_feats[None,None,:]
                #feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
            else:
                feats = ehr_feats[:,None,:]
                #feats = torch.cat([feats, cxr_feats[:,None,:]], dim=1)
        seq_lengths = np.array([1] * len(seq_lengths))
        #seq_lengths[pairs] = 2
        feats = torch.cat([feats, demo_feats], dim=1)
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)

        return {
            'lstm': fused_preds,
            'ehr_feats': ehr_feats,
            #'cxr_feats': cxr_feats,
            'demo_feats': demo_feats,
            #'note_feats': note_feats,
            }

    '''
    def forward_uni_ehr(self, x, seq_lengths=None, img=None ):
        ehr_preds , feats = self.ehr_model(x, seq_lengths)
        return {
            'uni_ehr': ehr_preds,
            'ehr_feats': feats
            }

    def forward_fused(self, x, seq_lengths=None, img=None, pairs=None ):

        ehr_preds , ehr_feats = self.ehr_model(x, seq_lengths)
        cxr_preds, _ , cxr_feats = self.cxr_model(img)
        projected = self.projection(cxr_feats)

        # loss = self.align_loss(projected, ehr_feats)

        feats = torch.cat([ehr_feats, projected], dim=1)
        fused_preds = self.fused_cls(feats)

        # late_avg = (cxr_preds + ehr_preds)/2
        return {
            'early': fused_preds, 
            'joint': fused_preds, 
            # 'late_avg': late_avg,
            # 'align_loss': loss,
            'ehr_feats': ehr_feats,
            'cxr_feats': projected,
            'unified': fused_preds
            }
    
    
    def forward_lstm_ehr(self, x, seq_lengths=None, img=None, pairs=None ):
        _ , ehr_feats = self.ehr_model(x, seq_lengths)
        feats = ehr_feats[:,None,:]
        
        
        seq_lengths = np.array([1] * len(seq_lengths))
        
        feats = torch.nn.utils.rnn.pack_padded_sequence(feats, seq_lengths, batch_first=True, enforce_sorted=False)

        x, (ht, _) = self.lstm_fusion_layer(feats)

        out = ht.squeeze()
        
        fused_preds = self.lstm_fused_cls(out)

        return {
            'uni_ehr_lstm': fused_preds,
        }

        
            
    def forward_uni_cxr(self, x, seq_lengths=None, img=None ):
        cxr_preds, _ , feats = self.cxr_model(img)
        return {
            'uni_cxr': cxr_preds,
            'cxr_feats': feats
            }
    '''