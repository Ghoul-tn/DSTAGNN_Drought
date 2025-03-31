#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numba import jit, prange
from tqdm import tqdm
import time

@jit(nopython=True)
def euclidean_distance(a, b):
    """Numba-compatible Euclidean distance"""
    return np.sqrt(np.sum((a - b)**2))

@jit(nopython=True, parallel=True)
def calculate_distances(coords, data_reduced, max_distance=10.0):
    """Fully Numba-optimized distance calculation"""
    n_nodes = coords.shape[0]
    sta_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in prange(n_nodes):
        for j in range(i+1, n_nodes):
            # Calculate spatial distance manually
            spatial_dist = euclidean_distance(coords[i], coords[j])
            
            if spatial_dist <= max_distance:
                # Calculate temporal similarity
                x = data_reduced[i]
                y = data_reduced[j]
                norm_x = np.sqrt(np.sum(x**2)) + 1e-12
                norm_y = np.sqrt(np.sum(y**2)) + 1e-12
                sta_matrix[i,j] = 1 - np.dot(x, y) / (norm_x * norm_y)
                
    return sta_matrix

def process_dataset(data_path, dataset_name, sparsity=0.01):
    print("Loading and preprocessing data...")
    with np.load(data_path) as f:
        data = f['data']  # (287, 2139, 4)
    
    # 1. Reduce temporal dimension with PCA
    pca = PCA(n_components=12)
    data_reshaped = data.transpose(1,0,2).reshape(2139, -1)  # (2139, 287*4)
    data_reduced = pca.fit_transform(data_reshaped)  # (2139, 12)
    
    print(f"PCA reduced dimensions from {data_reshaped.shape[1]} to {data_reduced.shape[1]}")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. Get spatial coordinates
    valid_pixels = ~np.isnan(data[0,:,0])
    coords = np.array(np.where(valid_pixels)).T  # (2139, 2)
    
    # 3. Calculate distances
    print("Calculating spatial-temporal distances...")
    start_time = time.time()
    sta_matrix = calculate_distances(coords, data_reduced)
    sta_matrix = sta_matrix + sta_matrix.T  # Symmetrize
    np.fill_diagonal(sta_matrix, 0)
    
    print(f"Distance calculation completed in {(time.time()-start_time)/60:.1f} minutes")
    
    # 4. Create adjacency matrices
    print("Creating adjacency matrices...")
    n_nodes = sta_matrix.shape[0]
    k = max(1, int(n_nodes * sparsity))
    
    A_adj = np.zeros_like(sta_matrix)
    R_adj = np.zeros_like(sta_matrix)
    
    for i in range(n_nodes):
        neighbors = np.argsort(sta_matrix[i])[:k]
        A_adj[i, neighbors] = 1
        R_adj[i, neighbors] = 1 - sta_matrix[i, neighbors]
    
    # Save results
    output_dir = os.path.dirname(data_path)
    
    np.save(os.path.join(output_dir, f"stag_001_{dataset_name}.npy"), sta_matrix)
    pd.DataFrame(A_adj).to_csv(os.path.join(output_dir, f"stag_001_{dataset_name}.csv"), header=False, index=False)
    pd.DataFrame(R_adj).to_csv(os.path.join(output_dir, f"strg_001_{dataset_name}.csv"), header=False, index=False)
    
    print("\nSuccessfully generated:")
    print(f"- stag_001_{dataset_name}.npy")
    print(f"- stag_001_{dataset_name}.csv")
    print(f"- strg_001_{dataset_name}.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .npz file path")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--sparsity", type=float, default=0.01, help="Connection sparsity")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Generating STA-Graph for {args.dataset}")
    print("="*60)
    process_dataset(args.input, args.dataset, args.sparsity)
