"""
Command-line interface for the VARLiNGAM causal discovery algorithm.

This script reads a time series from a CSV file, applies the VARLiNGAM model
to infer a causal graph, and exports the resulting structure.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
import os
from pathlib import Path

# --- Custom Module Imports ---
# VARLiNGAM: The main class for the VAR-LiNGAM algorithm from the 'lingam' library.
from lingam.var_lingam import VARLiNGAM

def calculate_summary_matrix(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Fits the VARLiNGAM model and calculates a summary matrix of causal influences.

    Args:
        data (np.ndarray): Input time series data, with shape (n_samples, n_variables).
        args (argparse.Namespace): Command-line arguments for model configuration.

    Returns:
        np.ndarray: A summary matrix where M[i, j] is the causal strength from node i to node j.
    """
    # 1. Instantiate and fit the VARLiNGAM model.
    # 'lags' sets the VAR order. 'prune=True' enables automatic model selection.
    model = VARLiNGAM(lags=args.max_lags, criterion='bic', prune=True)
    model.fit(data)

    # Check if the model fitting was successful.
    if model.adjacency_matrices_ is None:
        raise RuntimeError("VARLiNGAM fitting did not produce any adjacency matrices.")
    
    # 2. Process the raw coefficient matrices from the fitted model.
    # The result is a list of matrices, one for each lag.
    stacked_matrices = np.stack([np.abs(m) for m in model.adjacency_matrices_])

    # 3. Create a single summary graph of causal influences across all lags.
    # The convention for VARLiNGAM's adjacency_matrices_ is that B[i, j] represents
    # the influence of variable j on variable i (effect <- cause).
    # First, we take the maximum absolute influence across all lags (axis=0).
    # Then, we transpose the result to get the standard i -> j (cause -> effect) convention for graphs.
    summary_matrix = np.max(stacked_matrices, axis=0).transpose()

    # Ensure there are no self-loops in the final summary graph.
    np.fill_diagonal(summary_matrix, 0)

    return summary_matrix

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.
    """
    # --- Load Input Data ---
    try:
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
        headers = df.columns.tolist()
        data = df.to_numpy()
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Run Causal Discovery ---
    start_time = time.time()
    summary_matrix = calculate_summary_matrix(data, args)
    computation_time = time.time() - start_time

    # --- Save Computation Time ---
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # --- Create and Save the Output Graph ---
    G = nx.DiGraph()
    
    # First, add all variables as nodes to ensure any isolated nodes are included.
    for header in headers:
        G.add_node(header)
    
    # Add edges to the graph based on the summary matrix, applying a pruning threshold.
    # The matrix M[i, j] now represents the strength of the link i -> j.
    for i in range(len(headers)):  # From-node (cause)
        for j in range(len(headers)):  # To-node (effect)
            # Only add an edge if its weight is stronger than the specified threshold.
            if summary_matrix[i, j] > args.prune_threshold:
                weight = summary_matrix[i, j]
                G.add_edge(headers[i], headers[j], weight=weight)

    # --- Export Results ---
    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        # If no output file is specified, print the graph's adjacency list to the console.
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Parses command-line arguments and initiates the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Discover causal relationships in time series data using VARLiNGAM.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Group for I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', default=None, help='Input CSV file path. If not set, reads from standard input.')
    io_group.add_argument('-o', '--output', default=None, help='Output adjacency list file path. If not set, writes to standard output.')
    io_group.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', default=None, help='Output weighted edgelist file path.')
    
    # --- Group for VARLiNGAM Specific Arguments ---
    model_group = parser.add_argument_group('VARLiNGAM Specific Arguments')
    model_group.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to consider.')
    model_group.add_argument('--random_seed', type=int, default=42, help='Random seed for controlling reproducibility.')
    model_group.add_argument('--prune', type=bool, default=True, help='Enable adaptive lasso pruning for VARLiNGAM.')

    # --- Group for Output Filtering ---
    output_group = parser.add_argument_group('Output Filtering')
    output_group.add_argument('--prune_threshold', type=float, default=0, help='Pruning threshold for weak causal links in the learned model matrices.')

    args = parser.parse_args()
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()