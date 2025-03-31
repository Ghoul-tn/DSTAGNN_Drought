#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numba import jit, prange
from scipy.spatial.distance import cdist
from tqdm import tqdm
import time

@jit(nopython=True, parallel=True)
def calculate_distances(coords, data_reduced, max_distance=10.0):
    """Numba-optimized distance calculation for nearby nodes"""
    n_nodes = coords.shape[0]
    sta_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in prange(n_nodes):
        # Find nodes within spatial neighborhood
        distances = cdist(coords[i:i+1], coords)[0]
        neighbors = np.where(distances <= max_distance)[0]
        neighbors = neighbors[neighbors > i]  # Upper triangle only
        
        for j in neighbors:
            # Cosine distance between reduced features
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
    
    # 2. Get spatial coordinates from valid indices
    valid_pixels = ~np.isnan(data[0,:,0])  # Using NDVI for mask
    coords = np.array(np.where(valid_pixels)).T  # (2139, 2)
    
    # 3. Calculate distances (optimized)
    print("Calculating spatial-temporal distances...")
    start_time = time.time()
    sta_matrix = calculate_distances(coords, data_reduced)
    sta_matrix = sta_matrix + sta_matrix.T  # Symmetrize
    np.fill_diagonal(sta_matrix, 0)  # Zero diagonals
    
    print(f"Distance calculation completed in {(time.time()-start_time)/60:.1f} minutes")
    
    # 4. Create both adjacency matrices
    print("Creating adjacency matrices...")
    n_nodes = sta_matrix.shape[0]
    k = max(1, int(n_nodes * sparsity))
    
    # Binary adjacency (stag)
    A_adj = np.zeros_like(sta_matrix)
    # Weighted adjacency (strg)
    R_adj = np.zeros_like(sta_matrix)
    
    for i in range(n_nodes):
        neighbors = np.argsort(sta_matrix[i])[:k]
        A_adj[i, neighbors] = 1
        R_adj[i, neighbors] = 1 - sta_matrix[i, neighbors]  # Convert distance to similarity
    
    # Save results with original naming convention
    output_dir = os.path.dirname(data_path)
    
    # STA matrix (numpy format)
    np.save(os.path.join(output_dir, f"stag_001_{dataset_name}.npy"), sta_matrix)
    
    # Binary adjacency (csv)
    pd.DataFrame(A_adj).to_csv(
        os.path.join(output_dir, f"stag_001_{dataset_name}.csv"), 
        header=False, 
        index=False
    )
    
    # Weighted adjacency (csv)
    pd.DataFrame(R_adj).to_csv(
        os.path.join(output_dir, f"strg_001_{dataset_name}.csv"), 
        header=False, 
        index=False
    )
    
    print("\nGenerated files:")
    print(f"- stag_001_{dataset_name}.npy (STA matrix)")
    print(f"- stag_001_{dataset_name}.csv (binary adjacency)")
    print(f"- strg_001_{dataset_name}.csv (weighted adjacency)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate STA-graph files for drought prediction')
    parser.add_argument("--input", required=True, help="Path to input .npz file")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., GAMBIA)")
    parser.add_argument("--sparsity", type=float, default=0.01, help="Connection sparsity (0.01 = 1%)")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Generating STA-Graph for {args.dataset}")
    print("="*60)
    process_dataset(args.input, args.dataset, args.sparsity)
