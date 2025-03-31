#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import time
from scipy.optimize import linprog

def check_data(data, name):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError(f"Data contains NaN/Inf values: {name}")
    return data

def wasserstein_distance(p, q, D):
    """Calculate Wasserstein distance with robust handling"""
    A_eq = []
    size = len(p)
    
    # Row constraints
    for i in range(size):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    
    # Column constraints
    for j in range(size):
        A = np.zeros_like(D)
        A[:, j] = 1
        A_eq.append(A.reshape(-1))
    
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    
    # Clean numerical issues
    D = np.nan_to_num(D, nan=0.0, posinf=1e12, neginf=-1e12)
    A_eq = np.nan_to_num(A_eq, nan=0.0)
    b_eq = np.nan_to_num(b_eq, nan=0.0)
    
    result = linprog(D.reshape(-1), A_eq=A_eq, b_eq=b_eq)
    return result.fun

def spatial_temporal_aware_distance(x, y):
    """Calculate distance between two spatio-temporal series"""
    x, y = np.array(x), np.array(y)
    
    # Normalize
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    
    # Handle zeros
    x_norm[x_norm == 0] = 1e-12
    y_norm[y_norm == 0] = 1e-12
    
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    
    # Cosine distance matrix
    D = 1 - np.dot(x/x_norm, (y/y_norm).T)
    D = np.clip(D, 0, 1)  # Ensure valid distance
    
    return wasserstein_distance(p, q, D)

def process_dataset(data_path, period=12, sparsity=0.01):
    print("\nLoading drought data...")
    data = np.load(data_path)['data']  # Shape (287, 2139, 4)
    num_nodes = data.shape[1]
    
    print("Calculating spatial-temporal aware graph...")
    sta_matrix = np.zeros((num_nodes, num_nodes))
    
    # Process each node pair
    for i in range(num_nodes):
        if i % 100 == 0:
            print(f"Processing node {i}/{num_nodes}")
        
        for j in range(i+1, num_nodes):
            # Use all features (NDVI, SoilMoisture, LST, SPI)
            x = data[:, i, :]  # Shape (287, 4)
            y = data[:, j, :]  # Shape (287, 4)
            
            sta_matrix[i,j] = spatial_temporal_aware_distance(x, y)
    
    # Symmetrize the matrix
    sta_matrix = sta_matrix + sta_matrix.T
    
    # Save raw similarity matrix
    output_path = data_path.replace('.npz', '_sta.npy')
    np.save(output_path, sta_matrix)
    print(f"Saved raw STA matrix to {output_path}")
    
    # Create thresholded adjacency matrices
    print("\nGenerating adjacency matrices...")
    id_mat = np.identity(num_nodes)
    adj = 1 - sta_matrix + id_mat  # Convert similarity to distance
    
    # Create sparse adjacency matrix
    top = int(num_nodes * sparsity)
    A_adj = np.zeros_like(adj)
    
    for i in range(num_nodes):
        neighbors = np.argsort(adj[i,:])[:top]
        A_adj[i, neighbors] = 1
    
    # Save outputs
    adj_path = data_path.replace('.npz', '_adj.csv')
    np.savetxt(adj_path, A_adj, delimiter=',', fmt='%d')
    print(f"Saved adjacency matrix to {adj_path}")
    
    return sta_matrix, A_adj

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gambia_dstagnn.npz",
                       help="Path to processed drought data")
    parser.add_argument("--period", type=int, default=12,
                       help="Seasonal period (12 for monthly data)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                       help="Sparsity level for adjacency matrix")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Generating Spatial-Temporal Aware Graph for Drought Data")
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: 2139 | Timesteps: 287 | Features: 4")
    print("="*60)
    
    start_time = time.time()
    sta_matrix, adj_matrix = process_dataset(
        args.dataset,
        period=args.period,
        sparsity=args.sparsity
    )
    
    print(f"\nTotal processing time: {time.time()-start_time:.2f} seconds")
    print("Process completed successfully!")
