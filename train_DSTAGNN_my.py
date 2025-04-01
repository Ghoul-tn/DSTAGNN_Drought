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
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn

# Device setup
def setup_device():
    # Try TPU first
    if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print('Using TPU:', device)
            return device, 'tpu'
        except ImportError:
            print('TPU available but torch_xla not installed. Falling back to GPU/CPU.')
    
    # Try GPU next
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU:', device)
        return device, 'gpu'
    
    # Fallback to CPU
    device = torch.device('cpu')
    print('Using CPU')
    return device, 'cpu'

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Configuration file path")
    args = parser.parse_args()
    
    # Load config
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    
    # Setup device
    device, device_type = setup_device()
    
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
    
    # Load adjacency matrices
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
    
    # Memory optimizations
    if device_type == 'gpu':
        net = net.half()  # Mixed precision for GPU
        torch.backends.cudnn.benchmark = True
    
    # Training setup
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    criterion = nn.SmoothL1Loss().to(device)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(int(training_config['start_epoch']), int(training_config['epochs'])):
        # Train
        net.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if device_type == 'gpu':
                with torch.cuda.amp.autocast():
                    output = net(data)
                    loss = criterion(output, target)
                
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
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
            torch.save(net.state_dict(), f'best_model_epoch_{epoch}.pt')
            print(f'Saved new best model with val loss {best_val_loss:.4f}')

if __name__ == "__main__":
    main()
