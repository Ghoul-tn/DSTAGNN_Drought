#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import time
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return path

def wasserstein_distance(p, q, D):
    """Optimized Wasserstein distance calculation"""
    try:
        size = len(p)
        A_eq = np.zeros((2*size, size*size))
        
        # Row constraints
        for i in range(size):
            A_eq[i, i*size:(i+1)*size] = 1
        
        # Column constraints
        for j in range(size):
            A_eq[size+j, j::size] = 1
        
        b_eq = np.concatenate([p, q])
        D_clean = np.nan_to_num(D.reshape(-1), nan=0.0, posinf=1e12, neginf=-1e12)
        
        result = linprog(D_clean, A_eq=A_eq, b_eq=b_eq, method='highs')
        return result.fun if result.success else 1.0
    except:
        return 1.0

def process_node_pair(args):
    """Function to parallelize node pair processing"""
    i, j, data = args
    x = data[:, i, :]
    y = data[:, j, :]
    
    # Optimized distance calculation
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    x_norm[x_norm == 0] = 1e-12
    y_norm[y_norm == 0] = 1e-12
    
    p = x_norm[:, 0] / (x_norm.sum() + 1e-12)
    q = y_norm[:, 0] / (y_norm.sum() + 1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        D = 1 - np.dot(x/x_norm, (y/y_norm).T)
    D = np.nan_to_num(D, nan=1.0)
    D = np.clip(D, 0, 1)
    
    return (i, j, wasserstein_distance(p, q, D))

def process_dataset(data_path, dataset_name, period=12, sparsity=0.01):
    """Optimized main processing function"""
    try:
        # Validate and load data
        data_path = validate_path(data_path)
        dir_path = os.path.dirname(data_path)
        
        print(f"\nLoading data from: {data_path}")
        with np.load(data_path) as f:
            data = f['data']  # Shape (287, 2139, 4)
        num_nodes = data.shape[1]
        
        print(f"Data loaded. Shape: {data.shape}")
        print(f"Generating graph for {num_nodes} nodes using {cpu_count()} cores...")
        
        # Initialize matrix
        sta_matrix = np.zeros((num_nodes, num_nodes))
        
        # Prepare parallel processing
        start_time = time.time()
        tasks = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                tasks.append((i, j, data))
        
        # Process in parallel with progress bar
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            results = list(tqdm(executor.map(process_node_pair, tasks), 
                              total=len(tasks),
                              desc="Processing node pairs"))
        
        # Fill the matrix with results
        for i, j, dist in results:
            sta_matrix[i,j] = dist
        
        # Symmetrize and save
        sta_matrix = sta_matrix + sta_matrix.T
        sta_filename = f"stag_{int(sparsity*100):03d}_{dataset_name}.npy"
        sta_path = os.path.join(dir_path, sta_filename)
        np.save(sta_path, sta_matrix)
        print(f"\nSaved STA matrix to: {sta_path}")
        
        # Generate adjacency matrices (optimized)
        print("\nCreating adjacency matrices...")
        id_mat = np.identity(num_nodes)
        adj = 1 - sta_matrix + id_mat
        
        top = max(1, int(num_nodes * sparsity))
        A_adj = np.zeros_like(adj)
        R_adj = np.zeros_like(adj)
        
        # Parallelize this part too
        def process_row(i):
            neighbors = np.argsort(adj[i,:])[:top]
            A_adj[i, neighbors] = 1
            R_adj[i, neighbors] = adj[i, neighbors]
            return i
        
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(process_row, range(num_nodes)), 
                     total=num_nodes,
                     desc="Creating adjacency"))
        
        # Save with original naming
        adj_path = os.path.join(dir_path, f"stag_{int(sparsity*100):03d}_{dataset_name}.csv")
        strg_path = os.path.join(dir_path, f"strg_{int(sparsity*100):03d}_{dataset_name}.csv")
        
        pd.DataFrame(A_adj).to_csv(adj_path, header=False, index=False)
        pd.DataFrame(R_adj).to_csv(strg_path, header=False, index=False)
        
        print(f"Saved adjacency matrix to: {adj_path}")
        print(f"Saved weighted matrix to: {strg_path}")
        
        return sta_matrix, A_adj
        
    except Exception as e:
        print(f"\nError in processing: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimized STA Graph Generator')
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input .npz file")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., 'GAMBIA')")
    parser.add_argument("--period", type=int, default=12,
                       help="Seasonal period (12 for monthly data)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                       help="Sparsity level (0.01 = 1% connections)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Optimized Drought STA-Graph Generator")
    print("="*60)
    print(f"CPU Cores Available: {cpu_count()}")
    print(f"Input: {args.input}")
    print(f"Dataset Name: {args.dataset}")
    print(f"Parameters: period={args.period}, sparsity={args.sparsity}")
    
    start_time = time.time()
    try:
        sta_matrix, adj_matrix = process_dataset(
            args.input,
            args.dataset,
            period=args.period,
            sparsity=args.sparsity
        )
        total_time = (time.time() - start_time)/60
        print(f"\nTotal processing time: {total_time:.1f} minutes")
        print("\nSuccessfully generated:")
        print(f"- STA Matrix: {sta_matrix.shape}")
        print(f"- Adjacency Matrix: {adj_matrix.shape}")
    except Exception as e:
        print(f"\nFailed to generate graph: {str(e)}")
