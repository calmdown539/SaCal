import numpy as np
import sys

from arguments import args_parser
parser = args_parser()
args = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from datasets.dataloader_nocxr import get_multimodal_datasets
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
import datetime
import random
from trainers.fusion_trainer import FusionTrainer
from torch.utils.data import Dataset, DataLoader

path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

def read_timeseries(args):
    path = 'data/ehr/2_episode1_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)

# Pad the time series to the same length
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in ret]
    return np.array(ret), seq_length

def my_collate(batch):
    # Time series data  (When missing, use an all-zero vector with shape of [1,76])
    ehr = [item[0][-512:] if np.array_equal(item[0], None) is False else np.zeros((1,76)) for item in batch]
    ehr, ehr_length = pad_zeros(ehr)
    

    # CXR image data    (When missing, use an all-zero vector with shape of [3,224,224])
    #cxr = torch.stack([item[1] if item[1] != None else torch.zeros(3, 224, 224) for item in batch])

    # Note text data    (An empty string has been used to indicate modality missing)
    # note = [item[2] for item in batch]

    # Demographic data

    demo = [item[1] for item in batch]    
    # Label
    label = np.array([item[2] for item in batch]).reshape(len(batch),-1)

    return [ehr, ehr_length, demo, label]

def main():

    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')


    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, args.task)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    
    trainer = FusionTrainer(
        train_dl, 
        val_dl, 
        args,
        test_dl=test_dl
    )
    if args.mode == 'train':
        print("==> training")
        trainer.train()
    elif args.mode == 'eval':
        trainer.eval()
    else:
        raise ValueError("not Implementation for args.mode")

if __name__ == '__main__':
    main()

