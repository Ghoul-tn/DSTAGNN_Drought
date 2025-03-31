#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import time
from scipy.optimize import linprog

def validate_path(path):
    """Ensure input file exists"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return path

def wasserstein_distance(p, q, D):
    """Robust Wasserstein distance calculation"""
    try:
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
        return result.fun if result.success else 1.0
    except:
        return 1.0

def spatial_temporal_aware_distance(x, y):
    """Robust distance between spatio-temporal series"""
    try:
        x, y = np.array(x), np.array(y)
        if x.size == 0 or y.size == 0:
            return 1.0
        
        # Normalize with safety checks
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        x_norm[x_norm == 0] = 1e-12
        y_norm[y_norm == 0] = 1e-12
        
        p = x_norm[:, 0] / (x_norm.sum() + 1e-12)
        q = y_norm[:, 0] / (y_norm.sum() + 1e-12)
        
        # Cosine distance matrix with clipping
        with np.errstate(divide='ignore', invalid='ignore'):
            D = 1 - np.dot(x/x_norm, (y/y_norm).T)
        D = np.nan_to_num(D, nan=1.0)
        D = np.clip(D, 0, 1)
        
        return wasserstein_distance(p, q, D)
    except:
        return 1.0

def process_dataset(data_path, dataset_name, period=12, sparsity=0.01):
    """Main processing function with original output naming"""
    try:
        # Validate and load data
        data_path = validate_path(data_path)
        dir_path = os.path.dirname(data_path)
        
        print(f"\nLoading data from: {data_path}")
        with np.load(data_path) as f:
            data = f['data']  # Shape (287, 2139, 4)
        num_nodes = data.shape[1]
        
        print(f"Data loaded. Shape: {data.shape}")
        print(f"Generating graph for {num_nodes} nodes...")
        
        # Initialize matrix
        sta_matrix = np.zeros((num_nodes, num_nodes))
        
        # Process node pairs with progress tracking
        start_time = time.time()
        batch_size = 100
        for i in range(num_nodes):
            if i % batch_size == 0 or i == num_nodes - 1:
                elapsed = time.time() - start_time
                remaining = (num_nodes - i) * (elapsed/(i+1)) if i > 0 else 0
                print(f"Processed {i}/{num_nodes} nodes | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
            
            for j in range(i+1, num_nodes):
                sta_matrix[i,j] = spatial_temporal_aware_distance(
                    data[:, i, :], 
                    data[:, j, :]   
                )
        
        # Symmetrize and save (original naming convention)
        sta_matrix = sta_matrix + sta_matrix.T
        sta_filename = f"stag_{int(sparsity*100):03d}_{dataset_name}.npy"
        sta_path = os.path.join(dir_path, sta_filename)
        np.save(sta_path, sta_matrix)
        print(f"\nSaved STA matrix to: {sta_path}")
        
        # Generate adjacency matrix (original naming convention)
        print("\nCreating adjacency matrices...")
        id_mat = np.identity(num_nodes)
        adj = 1 - sta_matrix + id_mat
        
        top = max(1, int(num_nodes * sparsity))
        A_adj = np.zeros_like(adj)
        R_adj = np.zeros_like(adj)
        
        for i in range(num_nodes):
            neighbors = np.argsort(adj[i,:])[:top]
            A_adj[i, neighbors] = 1
            R_adj[i, neighbors] = adj[i, neighbors]
        
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
    parser = argparse.ArgumentParser(description='Generate STA Graph for Drought Data')
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input .npz file")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., 'PEMS04')")
    parser.add_argument("--period", type=int, default=12,
                       help="Seasonal period (12 for monthly data)")
    parser.add_argument("--sparsity", type=float, default=0.01,
                       help="Sparsity level (0.01 = 1% connections)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Drought STA-Graph Generator (Original Output Naming)")
    print("="*60)
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
        print(f"\nTotal processing time: {(time.time()-start_time)/60:.1f} minutes")
        print("\nSuccessfully generated:")
        print(f"- STA Matrix: {sta_filename}")
        print(f"- Adjacency Matrix: stag_{int(args.sparsity*100):03d}_{args.dataset}.csv")
        print(f"- Weighted Matrix: strg_{int(args.sparsity*100):03d}_{args.dataset}.csv")
    except Exception as e:
        print(f"\nFailed to generate graph: {str(e)}")
