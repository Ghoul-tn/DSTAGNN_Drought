#!/usr/bin/env python
# coding: utf-8
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
from model.DSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix, load_weighted_adjacency_matrix2, load_PA
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if xm.xrt_world_size() <= 1:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Initialize TPU
    device = xm.xla_device()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/PEMS04_dstagnn.conf', type=str,
                       help="configuration file path")
    args = parser.parse_args()
    
    # Load config
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    # Set seed
    seed_torch(1)

    # Load data - keep on CPU initially
    train_x_tensor, train_loader, train_target_tensor, val_x_tensor, val_loader, val_target_tensor, test_x_tensor, test_loader, test_target_tensor, mean, std = load_graphdata_channel1(
        data_config['graph_signal_matrix_filename'],
        int(training_config['num_of_hours']),
        int(training_config['num_of_days']),
        int(training_config['num_of_weeks']),
        'cpu',  # Load to CPU first
        int(training_config['batch_size']))
    # Check your data loader
    print(f"Train loader: {len(train_loader)} batches")
    sample = next(iter(train_loader))
    print(f"Sample batch shapes: {sample[0].shape}, {sample[1].shape}")
    # Load adjacency matrices on CPU
    if data_config['dataset_name'] in ['PEMS04', 'PEMS08', 'PEMS07', 'PEMS03']:
        adj_mx = get_adjacency_matrix2(data_config['adj_filename'], 
                                     int(data_config['num_of_vertices']), 
                                     id_filename=data_config.get('id_filename'))
    else:
        adj_mx = load_weighted_adjacency_matrix2(data_config['adj_filename'], 
                                               int(data_config['num_of_vertices']))
    
    adj_TMD = load_weighted_adjacency_matrix(data_config['stag_filename'], 
                                           int(data_config['num_of_vertices']))
    adj_pa = load_PA(data_config['strg_filename'])

    # Convert numpy arrays to torch tensors on CPU first
    adj_mx = torch.FloatTensor(adj_mx)
    adj_TMD = torch.FloatTensor(adj_TMD)
    adj_pa = torch.FloatTensor(adj_pa)

    # Model setup - move to TPU after initialization
    graph_use = training_config['graph']
    adj_merge = adj_mx if graph_use == 'G' else adj_TMD
    
    # Initialize model on CPU first
    net = make_model(
        'cpu',  # Initialize on CPU
        int(training_config['in_channels']),
        int(training_config['nb_block']),
        int(training_config['in_channels']),
        int(training_config['K']),
        int(training_config['nb_chev_filter']),
        int(training_config['nb_time_filter']),
        1,  # time_strides
        adj_merge,
        adj_pa,
        adj_TMD,
        int(data_config['num_for_predict']),
        int(data_config['len_input']),
        int(data_config['num_of_vertices']),
        int(training_config['d_model']),
        int(training_config['d_k']),
        int(training_config['d_k']),  # d_v = d_k
        int(training_config['n_heads'])
    )
    
    # Move model and data to TPU device
    net = net.to(device)
    train_target_tensor = train_target_tensor.to(device)
    val_target_tensor = val_target_tensor.to(device)
    test_target_tensor = test_target_tensor.to(device)
    
    # Convert to parallel loaders
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    test_loader = pl.MpDeviceLoader(test_loader, device)

    # Training setup
    folder_dir = '{}_{}h{}d{}w_channel{}_{}'.format(
        training_config['model_name'],
        training_config['num_of_hours'],
        training_config['num_of_days'],
        training_config['num_of_weeks'],
        training_config['in_channels'],
        float(training_config['learning_rate']))
    
    params_path = os.path.join('myexperiments', data_config['dataset_name'], folder_dir)
    if xm.is_master_ordinal():
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        print(f'Params path: {params_path}')

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = int(training_config['start_epoch'])
    epochs = int(training_config['epochs'])
    
    for epoch in range(start_epoch, epochs):
        # Train
        net.train()
        total_loss = 0.0
        start_time = time()
        
        for batch_idx, (encoder_inputs, labels) in enumerate(train_loader):
            xm.optimizer_step(optimizer, barrier=True)  # TPU sync
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} - Input shape: {encoder_inputs.shape}")
                print(f"Batch {batch_idx} - Output shape: {outputs.shape}")
                xm.master_print(f"Batch {batch_idx} completed")
            if batch_idx % 100 == 0:
                print(f"Memory usage: {xm.get_memory_info(device)}")
            optimizer.zero_grad()
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            
            total_loss += loss.item()
            if batch_idx % 100 == 0 and xm.is_master_ordinal():
                print(f'Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}')
        
        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for encoder_inputs, labels in val_loader:
                outputs = net(encoder_inputs)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if xm.is_master_ordinal():
            print(f'Epoch {epoch} Val Loss: {avg_val_loss:.4f} Time: {time()-start_time:.2f}s')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                xm.save(net.state_dict(), os.path.join(params_path, f'epoch_{epoch}.params'))
    
    # Final test on best model
    if xm.is_master_ordinal():
        net.load_state_dict(torch.load(os.path.join(params_path, f'epoch_{best_epoch}.params')))
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for encoder_inputs, labels in test_loader:
                outputs = net(encoder_inputs)
                test_loss += criterion(outputs, labels).item()
        print(f'Final Test Loss: {test_loss/len(test_loader):.4f}')

if __name__ == "__main__":
    # Kaggle-specific TPU initialization
    if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(main, nprocs=8, start_method='fork')
    else:
        # Fallback to single TPU core if not in Colab/Kaggle
        main()
