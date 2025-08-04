import sys
import os
from os import path as o
storage_path = o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../.."))
sys.path.append(storage_path)
from tqdm.auto import trange
from tqdm import tqdm
import torch
from torch import Tensor, sigmoid
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from multimodn.multimodn_phe import MultiModN
from multimodn.encoders import MIMIC_MLPEncoder
from multimodn.decoders import MLPDecoder
from multimodn.history import MultiModNHistory
#from datasets.mimic import MIMICDataset
from datasets import get_multimodal_datasets 
from pipelines import utils
import pickle as pkl
import importlib
import haim_api
importlib.reload(haim_api)
from haim_api import HAIMDecoder, HAIM
import argparse
from pipelines.utils import Discretizer, Normalizer
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import copy

def my_collate(batch):

    # Time series data  (When missing, use an all-zero vector with shape of [1,76])
    ehr = [item[0][-512:] if np.array_equal(item[0], None) is False else np.zeros((1,76)) for item in batch]
    ehr, ehr_length = pad_zeros(ehr)
    mask_ehr = np.array([1 if np.array_equal(item[0], None) is False else 0 for item in batch])     # Marks whether EHR is included
    ehr_length = [0 if mask_ehr[i] == 0 else ehr_length[i] for i in range(len(ehr_length))]  # Remove fictitious time series

    # CXR image data    (When missing, use an all-zero vector with shape of [3,224,224])
    cxr = torch.stack([item[1] if item[1] != None else torch.zeros(3, 224, 224) for item in batch])
    mask_cxr = np.array([1 if item[1] != None else 0 for item in batch])

    # Note text data    (An empty string has been used to indicate modality missing)
    note = [item[2] for item in batch]
    mask_note = np.array([1 if item[2] != '' else 0 for item in batch])

    # Demographic data

    demo = [item[3] for item in batch]
    # Label
    label = np.array([item[4] for item in batch]).reshape(len(batch),-1)

    return [ehr, cxr, note, demo, label]


# Pad the time series to the same length
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in ret]
    return np.array(ret), seq_length


def read_timeseries(path):
    path = 'data/ehr/10151556_episode1_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)


performance_metrics = ['f1', 'auc', 'accuracy', 'sensitivity', 'specificity', 'fpr', 'tpr', 'precision', 'recall', \
    'tn', 'fp', 'fn', 'tp', 'thr_roc', 'thr_pr']

hyperparameters = ['model', 'target', 'fold', 'miss_perc', 'seed', 'state_size', 'batch_size', 'encoder_hidd_units', 'decoder_hidd_units', 'dropout', 'epochs']

save_logs = hyperparameters + performance_metrics

source_names = ['de', 'vd',  'vmd', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']

source_size = [ 6, 1024, 1024, 99, 242, 110, 768, 768, 768]

source_dict = dict(zip(source_names, source_size))

def main(args):
    PIPELINE_NAME = utils.extract_pipeline_name(sys.argv[0])    
    criterion = '(auc + bac)'    
    results_directory = os.path.join(storage_path, 'nips', 'results')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)               
    
    model_type = PIPELINE_NAME + '_' + criterion    
    
    results_file_path = os.path.join(results_directory, model_type + '.csv') 
    
    sources = [ 'ehr', 'cxr', 'note']

    source_spec = '_'.join(sources) 

    #targets = ['in-hospital-mortality','decompensation', 'readmission']
    targets = ['phenotyping']

    pathologies = '_'.join(targets)
    
    # Hyperparameters    
    state_size = 50

    learning_rate = .001

    epochs =  10

    decoder_hidd_units =  32

    err_penalty = 1

    state_change_penalty =  0

    dropout = 0.2

    batch_size_train = 64

    batch_size_val = batch_size_train    

    encoder_hidd_units = decoder_hidd_units
    
    miss_perc = 0

    nfold = 5    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discretizer = Discretizer(timestep=float(args.timestep), store_masks=True, impute_strategy='previous', start_time='zero')
    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    mutli_train_dl = []
    mutli_val_dl = []
    mutli_test_dl = []
    for target in targets:    
        print(f'Task : {target}')
        model_spec = target            
        # Dataset splitting based on hospitalisation id & aggregated label, i.e. samples with the same haim_id should be all either in train or validation or test subsets

        normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            normalizer_state = '/data2/linfx/FlexCare-main/normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(1.0)
            normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
        normalizer.load_params(normalizer_state)

        train_ds, val_ds, test_ds = get_multimodal_datasets(discretizer, normalizer, args, target)
        train_loader = DataLoader(train_ds, batch_size_train, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size_train, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size_train, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)



        seed = 0       
        torch.manual_seed(seed)            
        ex_prefix = f'seed_{seed}_state_size_{state_size}_batch_size_{batch_size_train}_dec_hidd_units_{decoder_hidd_units}_dropout_{dropout}'
        part_of_hyperparameters = [target, miss_perc, seed, state_size, batch_size_train, encoder_hidd_units, decoder_hidd_units, dropout, epochs]  
                     
            
        partitions = [76, 128, 768, 5]
        # ModN model specification
        encoders = [MIMIC_MLPEncoder(state_size, partition, (encoder_hidd_units, encoder_hidd_units), activation = F.relu, dropout = dropout, ) for partition in partitions]
        decoders = [MLPDecoder(state_size, (decoder_hidd_units, decoder_hidd_units ), 25, output_activation = sigmoid) for _ in [target]]
        model_modn =  MultiModN(state_size, encoders, decoders, err_penalty, state_change_penalty) 

        optimizer = torch.optim.Adam(list(model_modn.parameters()), learning_rate)

        criterion = nn.BCELoss()
        #criterion = nn.CrossEntropyLoss()

        history =  MultiModNHistory([target])
            
        directory = os.path.join(storage_path, 'models', model_spec, source_spec)    

        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path_modn = os.path.join(directory, PIPELINE_NAME + f'_modn_model_{ex_prefix}.pkl')
        best_model_path_modn = os.path.join(directory, PIPELINE_NAME + f'_modn_best_model_{ex_prefix}.pt')
        
        # ModN training
        best_auc_bac_sum = 0
        for epoch in range(epochs):  
            print(f'Epoch: {epoch+1}')          
            if epoch == epochs - 1:
                train_buff_modn = model_modn.train_epoch(train_loader, optimizer, criterion, history, last_epoch = True)                      
            else:
                model_modn.train_epoch(train_loader, optimizer, criterion, history)
            val_buff_modn = model_modn.test(val_loader, criterion, history, tag='val')
            # Save the best model based on the sum of validation auroc and bac
            #auc_bac_sum =  val_buff_modn[0][1] + (val_buff_modn[0][3] + val_buff_modn[0][4]) / 2
            print(val_buff_modn[0][0],val_buff_modn[0][1])
            auc_bac_sum = val_buff_modn[0][1] + val_buff_modn[0][0]

            if auc_bac_sum > best_auc_bac_sum:                                        
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model_modn.state_dict(),                    
                    'auc_bac_val_cum': auc_bac_sum,
                    }, best_model_path_modn)  
                best_auc_bac_sum = auc_bac_sum
                val_buff_modn_best = val_buff_modn

        pkl.dump(model_modn, open(model_path_modn, 'wb'))

        directory = os.path.join(storage_path, 'history', model_spec, source_spec)                
        if not os.path.exists(directory):
            os.makedirs(directory)
        history_path = os.path.join(directory, PIPELINE_NAME + f'_history_{ex_prefix}.pkl')
        pkl.dump(history, open(history_path, 'wb'))

        directory = os.path.join(storage_path, 'plots', model_spec, source_spec)
        if not os.path.exists(directory):
                os.makedirs(directory)
        plot_path = os.path.join(directory, PIPELINE_NAME + f'_plot_{ex_prefix}.png')  
            
        targets_to_display = [target]
        history.plot(plot_path, targets_to_display, show_state_change=False)
        history.print_results()
        
        # ModN testing       
            
        checkpoint = torch.load(best_model_path_modn)  
        model_modn.load_state_dict(checkpoint['model_state_dict'])
        test_modn_best = model_modn.test(test_loader, criterion)
        print(test_modn_best[0])
        # results_modn_best = pd.DataFrame(columns=save_logs,)
        # test_modn_best_sngl = list(map(lambda metric: metric.numpy(), test_modn_best[0]))
        # row = ['modn'] + part_of_hyperparameters + test_modn_best_sngl
        # results_modn_best.loc[0] = row
        # if os.path.isfile(results_file_path):
        #     results_modn_best.to_csv(results_file_path, mode='a', index=False, header=False)
        # else:
        #     results_modn_best.to_csv(results_file_path, mode='w', index=False)            


            
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', type=str, help='Path to the data',
                        default='/data')
    parser.add_argument('--ehr_path', type=str, help='Path to the ehr data',
                        default='/data')
    parser.add_argument('--cxr_path', type=str, help='Path to the cxr data',
                        default='/data')

    parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
    parser.add_argument('--normalizer_state', type=str, default=None, help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')

    parser.add_argument('--epochs', type=int, default=10, help='number of chunks to train')
    parser.add_argument('--device', type=str, default="cuda", help='cuda:number or cpu')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers for dataloader')


    args = parser.parse_args()

    main(args)
