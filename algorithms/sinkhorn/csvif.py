"""
Command-line interface for the Sinkhorn causal discovery algorithm.

This script uses a custom PyTorch-based Permutation-SVAR model to analyze
a time series from a CSV file, infer a causal graph, and export the results.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time

# Import the main causal discovery class
from sinkhorn import Sinkhorn

def calculate_summary_matrix(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Calculates the summary matrix using the Sinkhorn causal discovery model.
    
    Args:
        data: Input time series data.
        args: Command-line arguments containing the model configuration.
        
    Returns:
        A matrix where element [i, j] represents the causal strength from node i to node j.
    """
    # 1. Instantiate the Sinkhorn model with parameters from the command line
    model = Sinkhorn(
        max_lags=args.max_lags,
        l1_inst_strength=args.l1_inst_strength,
        l1_lagged_strength=args.l1_lagged_strength,
        save_path=args.save_path,
    )

    # 2. Fit the model to the data
    model.fit(data, epochs=args.epochs, lr=args.lr)
    
    # 3. Create a summary graph of causal influences across all lags
    # This involves taking the maximum absolute coefficient for any link across all time lags.
    if not model.adjacency_matrices:
        raise RuntimeError("Model fitting did not produce any adjacency matrices.")

    # Stack the absolute values of the adjacency matrices for each lag
    stacked_matrices = np.stack([np.abs(m) for m in model.adjacency_matrices])
    
    # The model's raw output B[i, j] represents the effect from j -> i.
    # We transpose the matrix of maximums to get the standard i -> j graph convention.
    summary_matrix = np.max(stacked_matrices, axis=0).transpose()
    
    # Remove self-loops from the final summary
    np.fill_diagonal(summary_matrix, 0)
    
    return summary_matrix

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.
    """
    # Read data from the specified input file or standard input
    try:
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
        headers = df.columns.tolist()
        data = df.to_numpy()
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # Time the causal discovery process
    start_time = time.time()
    summary_matrix = calculate_summary_matrix(data, args)
    computation_time = time.time() - start_time
    
    # Save the computation time if a file path is provided
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Create a directed graph from the summary matrix
    G = nx.DiGraph()
    
    # First, add all variables as nodes to ensure isolated nodes are included
    for header in headers:
        G.add_node(header)
    
    # Then, add edges only if their causal strength exceeds the pruning threshold
    for i in range(len(headers)):
        for j in range(len(headers)):
            if abs(summary_matrix[i, j]) > args.prune_threshold:
                weight = summary_matrix[i, j]
                G.add_edge(headers[i], headers[j], weight=weight)

    # Export the graph to a weighted edgelist file if requested
    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)
        print(f"Saved the weighted causal structure to {args.weighted_edgelist}")

    # Export the graph to an adjacency list file or print to standard output
    if args.output:
        nx.write_adjlist(G, args.output)
        print(f"Saved the causal structure to {args.output}")
    else:
        # Default to writing the adjacency list to the console
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Parses command-line arguments and runs the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Discover causal relationships in time series data using the Sinkhorn model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', help='Input CSV file path. If not set, reads from standard input.')
    io_group.add_argument('-o', '--output', help='Output adjacency list file path. If not set, writes to standard output.')
    io_group.add_argument('-t', '--time', help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', help='Output weighted edgelist file path.')
    io_group.add_argument('--save_path', help='Path to save the trained PyTorch model state.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for controlling reproducibility.')
    
    # --- Model Configuration ---
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to consider.')
    model_group.add_argument('--epochs', type=int, default=6000, help='Number of training epochs.')
    model_group.add_argument('--lr', type=float, default=0.002, help='Learning rate for the optimizer.')
    model_group.add_argument('--l1_inst_strength', type=float, default=0.001, help='L1 regularization weight for instantaneous effects.')
    model_group.add_argument('--l1_lagged_strength', type=float, default=0.001, help='L1 regularization weight for lagged effects.')
    
    # --- Output Filtering ---
    output_group = parser.add_argument_group('Output Filtering')
    output_group.add_argument('--prune_threshold', type=float, default=0.01, help='Pruning threshold for weak causal links in the learned model matrices.')
    
    args = parser.parse_args()
    process_data(args)

if __name__ == '__main__':
    main()
    
