from __future__ import print_function, division 
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pickle
import pandas as pd
from PIL import Image
import argparse
from torch.cuda import amp

from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, cohen_kappa_score, mean_squared_error, f1_score
from sklearn import metrics
from models.modeling_irene import IRENE, CONFIGS
from tqdm import tqdm
import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import datetime
tk_lim = 40
ehr_win = 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

disease_list = ['COPD', 'Bronchiectasis', 'Pneumothorax', 'Pneumonia', 'ILD', 'Tuberculosis', 'Lung cancer', 'Pleural effusion']
cache_dir = "pretrained/biobert-base-cased-v1.2"
bert = AutoModel.from_pretrained(cache_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(cache_dir)

def my_metrics(yt, yp, task=None):
    if task in ['in-hospital-mortality', 'decompensation', 'readmission']:
        yt = yt.view(-1).detach().cpu().numpy()
        yp = yp.view(-1).detach().cpu().numpy()
        precision, recall, _, = precision_recall_curve(yt, yp)
        aupr = metrics.auc(recall, precision)
        auc = roc_auc_score(yt, yp)
        return auc, aupr
    elif task == 'phenotyping' or task == 'diagnosis':
        yt = yt.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        total_auc = 0.
        
        #print(yt.shape,yp.shape)
        for i in range(yt.shape[1]):
            label_mask = (yt[:, i] > -1)
            try:
                auc = roc_auc_score(yt[:, i][label_mask], yp[:, i][label_mask])
            except ValueError:
                auc = 0.5
            total_auc += auc
        macro_auc = total_auc/yt.shape[1]

        label_mask = (yt > -1)
        try:
            micro_auc = roc_auc_score(yt[label_mask], yp[label_mask])
        except ValueError:
            micro_auc = 0.5
        return macro_auc, micro_auc
    else:
        micro_F1 = f1_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy(), average="micro")
        macro_F1 = f1_score(yt.detach().cpu().numpy(), yp.detach().cpu().numpy(), average="macro")
        return micro_F1, macro_F1


def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading IRENE...")
    return model

def computeAUROC (dataGT, dataPRED, classCount=8):
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

def get_note_embedding(note, tokenizer, model, device = 'cpu'):
    """Convert clinical note to BERT embedding."""
    inputs = tokenizer(note, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token embedding
class Data(Dataset):
    def __init__(self, data_path, tokenizer, bert_model, transform=None, target_transform=None, device = 'cpu'):
        dict_path = data_path+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
    


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]

        label = self.mm_data[k]['label'].astype('float32')
        
        # note = self.mm_data[k]['pdesc']
        # note = get_note_embedding(note, self.tokenizer, self.bert_model, self.device)
        # note = torch.from_numpy(np.array(note)).float()
        # note = note.repeat(tk_lim, 1)
        #print(note.shape)

        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        ehr = torch.from_numpy(self.mm_data[k]['bts']).float()
        ehr_len = 48
        ehr = torch.from_numpy(self.mm_data[k]['bts']).float()

        if ehr.shape[0] >= ehr_len:
            ehr = ehr[:ehr_len]
        else:
            pad = torch.zeros(ehr_len - ehr.shape[0], ehr.shape[1])
            ehr = torch.cat([ehr, pad], dim=0)

        # if self.mm_data[k]['img'] == None:
        #     self.mm_data[k]['img'] = torch.zeros(3, 224, 224)
            
        #img = self.mm_data[k]['img'].float()
        #return img, label, note, ehr
        return label, demo, ehr

def test(args):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["IRENE"]
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes).to(device)
    task = args.TASK
    train_data = Data(f'{args.DATA_DIR}/{task}_train', tokenizer, bert, args.TASK, device=device)
    val_data = Data(f'{args.DATA_DIR}/{task}_val', tokenizer, bert, args.TASK, device=device)
    test_data = Data(f'{args.DATA_DIR}/{task}_test', tokenizer, bert, args.TASK, device=device)
    train_loader = DataLoader(train_data, batch_size=args.BSZ, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.BSZ, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=0, pin_memory=True)

    file_path = 'log/nocxr_lr_' + str(args.LR) + '_epoch_' + str(args.EPOCHS) +'_' + str(args.TASK)+ '_' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.txt'
    if not os.path.exists('log'):
        os.mkdir('log')

    # with open(file_path, "a", encoding='utf-8') as f:
    #     f.write("fusion moe\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=0.01)
    #model, optimizer = amp.initialize(model.cuda(), optimizer, opt_level="O1")

    model = torch.nn.DataParallel(model)
    scaler = amp.GradScaler()
    best_val_auroc = 0.0
    best_model_path = f'model_nocxr/{args.TASK}_best_model.pth'
    
    # print('--------Start training-------')
    for epoch in range(args.EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.EPOCHS}"):
            label, demo, ehr= data

            # note = note.view(-1, tk_lim, note.shape[-1]).to(device, non_blocking=True).float()
            # if torch.isnan(note).any() or torch.isinf(note).any():
            #     print("note is NaN or Inf!")
            #     note = torch.nan_to_num(note)
            demo = demo.view(-1, 1, demo.shape[-1]).to(device, non_blocking=True).float()
            if torch.isnan(demo).any() or torch.isinf(demo).any():
                print("demo is NaN or Inf!")
                demo = torch.nan_to_num(demo)
            ehr = ehr.view(-1, ehr_win, ehr.shape[-1]).to(device, non_blocking=True).float()
            ehr = ehr.mean(dim=1, keepdim=True)
            if torch.isnan(ehr).any() or torch.isinf(ehr).any():
                print("ehr is NaN or Inf!")
                ehr = torch.nan_to_num(ehr)

            marital_status = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            race = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            insurance = demo[:, :, 2].view(-1, 1, 1).cuda(non_blocking=True).float()
            #age = demo[:, :, 4].view(-1, 1, 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 3].view(-1, 1, 1).cuda(non_blocking=True).float()
            
            # img = img.to(device, non_blocking=True)
            # if torch.isnan(img).any() or torch.isinf(img).any():
            #     print("img is NaN or Inf!")
            #     img = torch.nan_to_num(img)
            label = label.to(device, non_blocking=True)
            if torch.isnan(label).any() or torch.isinf(label).any():
                print("label is NaN or Inf!")
                label = torch.nan_to_num(label)


            optimizer.zero_grad()
            loss, _, _ = model(None, None, ehr, sex, None, race, marital_status, insurance, labels=label, task=args.TASK)

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            # optimizer.step()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}")
        with open(file_path, "a", encoding='utf-8') as f:
            f.write('Epoch:' + str(epoch) + '\n')
            f.write('lr:'+str(args.LR)+ '\n')
            f.write('Train loss:'+str(running_loss)+'\n')

        # Validation
        model.eval()
        val_loss = 0.0
        outGT = torch.FloatTensor().to(device, non_blocking=True)
        outPRED = torch.FloatTensor().to(device, non_blocking=True)
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                label, demo, ehr= data
                # note = note.view(-1, tk_lim, note.shape[-1]).to(device, non_blocking=True).float()
                # note = torch.nan_to_num(note)
                demo = demo.view(-1, 1, demo.shape[-1]).to(device, non_blocking=True).float()
                demo = torch.nan_to_num(demo)
                ehr = ehr.view(-1, ehr_win, ehr.shape[-1]).to(device, non_blocking=True).float()
                ehr = ehr.mean(dim=1, keepdim=True)
                ehr = torch.nan_to_num(ehr)
                marital_status = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                race = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
                insurance = demo[:, :, 2].view(-1, 1, 1).cuda(non_blocking=True).float()
                #age = demo[:, :, 4].view(-1, 1, 1).cuda(non_blocking=True).float()
                sex = demo[:, :, 3].view(-1, 1, 1).cuda(non_blocking=True).float()
                #img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)


                loss, logits, _ = model(None, None, ehr, sex, None, race, marital_status, insurance, labels=label, task=args.TASK)

                val_loss += loss.item()
                #probs = torch.sigmoid(logits) if args.TASK != 'length-of-stay' else torch.softmax(logits, dim=-1)
                if task == 'length-of-stay':
                    label = label.long().view(-1)
                    _ , logits = torch.max(logits, dim=1)
                probs = logits
                outGT = torch.cat((outGT, label), 0)
                outPRED = torch.cat((outPRED, probs), 0)
                

        val_loss /= len(val_loader)

        auc, aupr = my_metrics(outGT, outPRED, args.TASK)
        valid_res = auc + aupr
        print(f"Epoch {epoch+1}/{args.EPOCHS}, Val Loss: {val_loss:.4f}, Val Accuracy: {auc:.4f}")
        with open(file_path, "a", encoding='utf-8') as f:
            f.write('Valid loss:' + str(val_loss)+'Valid auc:' + str(valid_res)+'\n')

        if valid_res > best_val_auroc:
            best_val_auroc = valid_res
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    #----- Test ------
    print('--------Start testing-------')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    outGT = torch.FloatTensor().to(device, non_blocking=True)
    outPRED = torch.FloatTensor().to(device, non_blocking=True)
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            label, demo, ehr= data
            # note = note.view(-1, tk_lim, note.shape[-1]).to(device, non_blocking=True).float()
            # note = torch.nan_to_num(note)
            demo = demo.view(-1, 1, demo.shape[-1]).to(device, non_blocking=True).float()
            demo = torch.nan_to_num(demo)
            ehr = ehr.view(-1, ehr_win, ehr.shape[-1]).to(device, non_blocking=True).float()
            ehr = ehr.mean(dim=1, keepdim=True)
            ehr = torch.nan_to_num(ehr)
            marital_status = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            race = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            insurance = demo[:, :, 2].view(-1, 1, 1).cuda(non_blocking=True).float()
            #age = demo[:, :, 4].view(-1, 1, 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 3].view(-1, 1, 1).cuda(non_blocking=True).float()
            #img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if args.TASK == 'length-of-stay':
                label = label.long().view(-1)

            logits, _, _ = model(None, None, ehr, sex, None, race, marital_status, insurance, task=args.TASK)
            #probs = torch.sigmoid(logits) if args.TASK != 'length-of-stay' else torch.softmax(logits, dim=-1)
            if task == 'length-of-stay':
                label = label.long().view(-1)
                _ , logits = torch.max(logits, dim=1)
            #print(logits)
            probs = logits
            # print(probs)
            # print(label,outGT)
            outGT = torch.cat((outGT, label), 0)
            outPRED = torch.cat((outPRED, probs), 0)

    auc, aupr = my_metrics(outGT, outPRED, args.TASK)
    print(f"Test Accuracy: {auc:.4f}, AUPR: {aupr:.4f}")
    with open(file_path, "a", encoding='utf-8') as f:
        f.write('Test AUC:'+str(auc)+'   AUPR:'+str(aupr) + '\n')

         
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--CLS', type=int, default=1, help='Number of classes for diagnosis task')
    parser.add_argument('--BSZ', type=int, default=64, help='Batch size')
    parser.add_argument('--DATA_DIR', type=str, default = 'data', help='')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--LR', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--TASK', type=str, choices=['in-hospital-mortality', 'decompensation', 'length-of-stay','readmission','phenotyping'], required=True, help='Task to perform')
    args = parser.parse_args()
    test(args)
