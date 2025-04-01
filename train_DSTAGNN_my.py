#!/usr/bin/env python
import os
import torch
import argparse
import configparser
from time import time
from model.DSTAGNN_my import make_model
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2, compute_val_loss_mstgcn, predict_and_save_results_mstgcn

# TPU/GPU detection and setup
def setup_device():
    if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f'Using TPU: {device}')
            return device, 'tpu'
        except ImportError:
            print('TPU available but torch_xla not installed. Falling back to GPU/CPU.')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {device}')
        return device, 'gpu'
    
    device = torch.device('cpu')
    print('Using CPU')
    return device, 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Configuration file path")
    args = parser.parse_args()
    
    # Setup device
    device, device_type = setup_device()
    
    # Load config
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    
    # Load data
    train_x_tensor, train_loader, train_target_tensor, val_x_tensor, val_loader, val_target_tensor, test_x_tensor, test_loader, test_target_tensor, mean, std = load_graphdata_channel1(
        data_config['graph_signal_matrix_filename'],
        int(training_config['num_of_hours']),
        int(training_config['num_of_days']),
        int(training_config['num_of_weeks']),
        device,
        int(training_config['batch_size']),
        shuffle=True
    )
    
    # Initialize model
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
    ).to(device)
    
    # Memory optimizations
    if device_type == 'tpu':
        net = torch.compile(net)  # XLA compilation for TPU
    elif device_type == 'gpu':
        net = torch.compile(net, mode='reduce-overhead')  # GPU optimization
        net.half()  # Mixed precision for GPU
    
    # Training setup
    optimizer = optim.Adam(net.parameters(), lr=float(training_config['learning_rate']))
    criterion = nn.SmoothL1Loss().to(device)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(int(training_config['start_epoch']), int(training_config['epochs'])):
        train_loss = train_epoch(net, train_loader, optimizer, criterion, device, device_type)
        val_loss = validate(net, val_loader, criterion, device)
        
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(net, epoch, val_loss, device_type)

def train_epoch(net, train_loader, optimizer, criterion, device, device_type):
    net.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision for GPU
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
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(net, val_loader, criterion, device):
    net.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def save_checkpoint(net, epoch, val_loss, device_type):
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'val_loss': val_loss,
    }
    
    # Use new zipfile serialization for smaller files
    torch.save(state, f'checkpoint_epoch_{epoch}.pt', _use_new_zipfile_serialization=True)
    print(f'Saved checkpoint at epoch {epoch} with val loss {val_loss:.4f}')

if __name__ == "__main__":
    main()
