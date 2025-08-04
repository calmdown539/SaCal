# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.cc_len
        num_ehr = config.ehr_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.cc_embeddings = Linear(768, config.hidden_size)  # text
        self.ehr_embeddings = Linear(76, config.hidden_size)  #ehr

        self.sex_embeddings = Linear(1, config.hidden_size)  
        self.age_embeddings = Linear(1, config.hidden_size)  
        self.race_embeddings = Linear(1, config.hidden_size)
        self.insurance_embeddings = Linear(1, config.hidden_size)
        self.marital_status_embeddings = Linear(1, config.hidden_size)  
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+n_patches, config.hidden_size))
        self.pe_cc = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
        self.pe_ehr = nn.Parameter(torch.zeros(1, num_ehr, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_race = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_marital_status = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_insurance = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_cc = Dropout(config.transformer["dropout_rate"])
        self.dropout_ehr = Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = Dropout(config.transformer["dropout_rate"])
        self.dropout_age = Dropout(config.transformer["dropout_rate"])
        self.dropout_race = Dropout(config.transformer["dropout_rate"])
        self.dropout_marital_status = Dropout(config.transformer["dropout_rate"])
        self.dropout_insurance = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, cc, ehr, sex=None, age=None, race=None, marital_status=None, insurance=None):
        B = ehr.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # if self.hybrid:
        #     x = self.hybrid_model(x)
        # x = self.patch_embeddings(x) # 16*16 --> CNN --> 1*1
        # cc = self.cc_embeddings(cc)
        # ehr = self.ehr_embeddings(ehr)
        # sex = self.sex_embeddings(sex)
        # age = self.age_embeddings(age)
        # race = self.race_embeddings(race)
        # marital_status = self.marital_status_embeddings(marital_status)
        # insurance = self.insurance_embeddings(insurance)
        def safe_tensor(t, name="tensor"):
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"[Embeddings] Warning: {name} contains NaN or Inf")
                t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            t = torch.clamp(t, min=-1e4, max=1e4)
            return t

        #cc = safe_tensor(self.cc_embeddings(cc), "cc embedding")
        ehr = safe_tensor(self.ehr_embeddings(ehr), "ehr embedding")
        sex = safe_tensor(self.sex_embeddings(sex), "sex embedding")
        #age = safe_tensor(self.age_embeddings(age), "age embedding")
        race = safe_tensor(self.race_embeddings(race), "race embedding")
        marital_status = safe_tensor(self.marital_status_embeddings(marital_status), "marital_status embedding")
        insurance = safe_tensor(self.insurance_embeddings(insurance), "insurance embedding")

        # x = x.flatten(2)
        # x = x.transpose(-1, -2)
        # x = torch.cat((cls_tokens, x), dim=1)

        # embeddings = x + self.position_embeddings
        #cc_embeddings = cc + self.pe_cc
        ehr_embeddings = ehr + self.pe_ehr
        sex_embeddings = sex + self.pe_sex
        #age_embeddings = age + self.pe_age
        race_embeddings = race + self.pe_race
        marital_status_embeddings = marital_status + self.pe_marital_status
        insurance_embeddings = insurance + self.pe_insurance

        #embeddings = self.dropout(embeddings)
        #cc_embeddings = self.dropout_cc(cc_embeddings)
        ehr_embeddings = self.dropout_ehr(ehr_embeddings)
        sex_embeddings = self.dropout_sex(sex_embeddings)
        #age_embeddings = self.dropout_age(age_embeddings)
        race_embeddings = self.dropout_race(race_embeddings)
        marital_status_embeddings = self.dropout_marital_status(marital_status_embeddings)
        insurance_embeddings = self.dropout_insurance(insurance_embeddings)
        #return embeddings, cc_embeddings, ehr_embeddings, sex_embeddings, age_embeddings, race_embeddings, marital_status_embeddings, insurance_embeddings
        #return cc_embeddings, ehr_embeddings, sex_embeddings, age_embeddings, race_embeddings, marital_status_embeddings, insurance_embeddings
        return ehr_embeddings, sex_embeddings, race_embeddings, marital_status_embeddings, insurance_embeddings


