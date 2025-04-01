#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from time import time
import shutil
import argparse
import configparser

# TPU imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from model.DSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix, load_weighted_adjacency_matrix2, load_PA
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(1)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/GAMBIA_dstagnn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# Device configuration
use_tpu = False
if 'use_tpu' in training_config and training_config['use_tpu'] == 'True':
    use_tpu = True
    device = xm.xla_device()
    print("Using TPU")
else:
    if training_config['ctx'] == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

# Model configuration
adj_mx = get_adjacency_matrix2(data_config['adj_filename'], int(data_config['num_of_vertices']), 
                              id_filename=data_config.get('id_filename', None))
adj_TMD = load_weighted_adjacency_matrix(data_config['stag_filename'], int(data_config['num_of_vertices']))
adj_pa = load_PA(data_config['strg_filename'])

if training_config['graph'] == 'G':
    adj_merge = adj_mx
else:
    adj_merge = adj_TMD

net = make_model(device, 
                int(training_config['num_of_d']),
                int(training_config['nb_block']),
                int(training_config['in_channels']),
                int(training_config['K']),
                int(training_config['nb_chev_filter']),
                int(training_config['nb_time_filter']),
                int(training_config['time_strides']),
                adj_merge,
                adj_pa,
                adj_TMD,
                int(data_config['num_for_predict']),
                int(data_config['len_input']),
                int(data_config['num_of_vertices']),
                int(training_config['d_model']),
                int(training_config['d_k']),
                int(training_config['d_v']),
                int(training_config['n_heads']))

def train_main():
    # Create output directory
    params_path = os.path.join('myexperiments', data_config['dataset_name'], 
                             f"{training_config['model_name']}_h{training_config['num_of_hours']}"
                             f"d{training_config['num_of_days']}w{training_config['num_of_weeks']}"
                             f"_channel{training_config['in_channels']}_{float(training_config['learning_rate'])}")
    
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))

    # Load data
    train_x_tensor, train_loader, train_target_tensor, \
    val_x_tensor, val_loader, val_target_tensor, \
    test_x_tensor, test_loader, test_target_tensor, \
    _mean, _std = load_graphdata_channel1(
        data_config['graph_signal_matrix_filename'],
        int(training_config['num_of_hours']),
        int(training_config['num_of_days']),
        int(training_config['num_of_weeks']),
        device,
        int(training_config['batch_size']))

    # Convert to TPU parallel loaders if using TPU
    if use_tpu:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    # Training setup
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    
    # Gradient accumulation steps
    accumulation_steps = 4 if use_tpu else 1
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(int(training_config['start_epoch']), int(training_config['epochs'])):
        net.train()
        optimizer.zero_grad()
        
        for batch_idx, (encoder_inputs, labels) in enumerate(train_loader):
            encoder_inputs, labels = encoder_inputs.to(device), labels.to(device)
            
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_tpu:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}')
                    sw.add_scalar('training_loss', loss.item() * accumulation_steps, global_step)
        
        # Validation
        val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch)
        print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(params_path, f'epoch_{epoch}.params')
            if use_tpu:
                xm.save(net.state_dict(), model_path)
            else:
                torch.save(net.state_dict(), model_path)
            print(f'Saved best model at epoch {epoch} with val loss {val_loss:.4f}')
    
    # Final testing
    predict_and_save_results_mstgcn(net, test_loader, test_target_tensor, 
                                  global_step, _mean, _std, params_path, 'test')

if __name__ == "__main__":
    train_main()
