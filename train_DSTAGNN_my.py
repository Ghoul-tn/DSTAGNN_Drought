#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import configparser
from time import time
from model.DSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix, load_weighted_adjacency_matrix2, load_PA
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2

def setup_tpu():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print('Using TPU:', device)
        return device, True
    except ImportError:
        print('torch_xla not available. Falling back to CPU/GPU.')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu'), False

def convert_adjacency_matrix(matrix, device):
    """Convert adjacency matrix to proper tensor format for the target device"""
    if isinstance(matrix, np.ndarray):
        matrix = torch.FloatTensor(matrix)
    elif isinstance(matrix, torch.Tensor):
        matrix = matrix.float()
    else:
        raise ValueError("Unsupported matrix type")
    return matrix.to(device)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Configuration file path")
    args = parser.parse_args()
    
    # Setup device
    device, is_tpu = setup_tpu()
    
    # Load config
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    
    # Load data
    print("Loading data...")
    train_x_tensor, train_loader, train_target_tensor, val_x_tensor, val_loader, val_target_tensor, test_x_tensor, test_loader, test_target_tensor, mean, std  = load_graphdata_channel1(
        data_config['graph_signal_matrix_filename'],
        int(training_config['num_of_hours']),
        int(training_config['num_of_days']),
        int(training_config['num_of_weeks']),
        device,
        int(training_config['batch_size']),
        shuffle=True
    )
    
    # Load and convert adjacency matrices
    print("Loading adjacency matrices...")
    adj_mx = get_adjacency_matrix2(
        data_config['adj_filename'], 
        int(data_config['num_of_vertices']),
        id_filename=data_config.get('id_filename')
    )
    
    adj_TMD = load_weighted_adjacency_matrix(
        data_config['stag_filename'], 
        int(data_config['num_of_vertices'])
    )
    
    adj_pa = load_PA(data_config['strg_filename'])
    
    # Convert to tensors and move to device
    adj_mx = convert_adjacency_matrix(adj_mx, device)
    adj_TMD = convert_adjacency_matrix(adj_TMD, device)
    adj_pa = convert_adjacency_matrix(adj_pa, device)
    
    # Initialize model
    print("Initializing model...")
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
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    criterion = nn.SmoothL1Loss().to(device)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(int(training_config['start_epoch']), int(training_config['epochs'])):
        # Train
        if str(device) == 'xla':
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            xm.mark_step()  # Important for TPU execution
        net.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            
            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
            else:
                optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}')
        
        # Validate
        val_loss = 0
        net.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.save(net.state_dict(), f'best_model_epoch_{epoch}.pt')
            else:
                torch.save(net.state_dict(), f'best_model_epoch_{epoch}.pt')
            print(f'Saved new best model with val loss {best_val_loss:.4f}')

if __name__ == "__main__":
    main()
