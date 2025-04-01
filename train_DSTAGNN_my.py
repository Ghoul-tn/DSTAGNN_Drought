#!/usr/bin/env python
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from model.DSTAGNN_my import make_model
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn

def setup_tpu():
    """Initialize TPU settings"""
    device = xm.xla_device()
    xm.rendezvous("init")
    return device

def train_fn(index, args, flgs):
    """Main training function for TPU core"""
    # Initialize TPU
    device = setup_tpu()
    torch.manual_seed(42 + index)
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    
    # Load data
    train_loader, val_loader, test_loader, mean, std = load_data(data_config, training_config, device)
    
    # Initialize model
    net = initialize_model(data_config, training_config, device)
    net.to(device)
    
    # Training setup
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    criterion = nn.SmoothL1Loss().to(device)
    
    # Parallel loader
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    
    # Training loop
    for epoch in range(int(training_config['start_epoch']), int(training_config['epochs'])):
        train_epoch(net, train_loader, optimizer, criterion, epoch, device)
        val_loss = validate(net, val_loader, criterion, epoch, device)
        
        # Save checkpoint
        if xm.is_master_ordinal() and val_loss < best_val_loss:
            save_checkpoint(net, epoch, val_loss)

def load_data(data_config, training_config, device):
    """Load and prepare data with memory optimizations"""
    # Use float16 to reduce memory
    torch.set_default_dtype(torch.float16)
    
    train_loader, train_target, val_loader, val_target, test_loader, test_target, mean, std = load_graphdata_channel1(
        data_config['graph_signal_matrix_filename'],
        int(training_config['num_of_hours']),
        int(training_config['num_of_days']),
        int(training_config['num_of_weeks']),
        device,
        int(training_config['batch_size']),
        shuffle=True
    )
    
    # Convert to float16
    mean = mean.half()
    std = std.half()
    
    return train_loader, val_loader, test_loader, mean, std

def initialize_model(data_config, training_config, device):
    """Initialize model with memory optimizations"""
    adj_mx = get_adjacency_matrix2(data_config['adj_filename'], 
                                  int(data_config['num_of_vertices']),
                                  id_filename=data_config.get('id_filename'))
    
    adj_TMD = load_weighted_adjacency_matrix(data_config['stag_filename'], 
                                           int(data_config['num_of_vertices']))
    adj_pa = load_PA(data_config['strg_filename'])
    
    net = make_model(
        device,
        int(training_config['in_channels']),
        int(training_config['nb_block']),
        int(training_config['in_channels']),
        int(training_config['K']),
        int(training_config['nb_chev_filter']),
        int(training_config['nb_time_filter']),
        1,  # time_strides
        adj_mx if training_config['graph'] == 'G' else adj_TMD,
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
    
    # Apply memory optimizations
    net.half()  # Convert model to float16
    return net

def train_epoch(net, train_loader, optimizer, criterion, epoch, device):
    """Training loop with memory optimizations"""
    net.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            output = net(data)
            loss = criterion(output, target)
        
        # Gradient scaling for float16
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Synchronize and report
        loss_value = loss.item()
        xm.master_print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss_value:.4f}')
        total_loss += loss_value
    
    avg_loss = total_loss / len(train_loader)
    xm.master_print(f'Epoch: {epoch} | Training Avg Loss: {avg_loss:.4f}')

def validate(net, val_loader, criterion, epoch, device):
    """Validation loop"""
    net.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    xm.master_print(f'Epoch: {epoch} | Validation Loss: {avg_loss:.4f}')
    return avg_loss

def save_checkpoint(net, epoch, val_loss):
    """Save model checkpoint with memory optimizations"""
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'val_loss': val_loss,
    }
    
    # Save with reduced precision
    torch.save(state, f'checkpoint_epoch_{epoch}.pt', _use_new_zipfile_serialization=True)
    xm.master_print(f'Saved checkpoint at epoch {epoch}')

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Configuration file path")
    args = parser.parse_args()
    
    # TPU training
    xmp.spawn(train_fn, args=(args, {}), nprocs=8, start_method='fork')
