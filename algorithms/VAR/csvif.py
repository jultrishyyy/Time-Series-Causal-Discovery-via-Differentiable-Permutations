"""
Command-line interface for causal discovery using a Vector Autoregression (VAR) model.

This script reads a time series from a CSV file, fits a VAR model to infer
causal influences, and exports the resulting weighted directed graph.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time

# --- Third-Party Imports ---
# VAR: The Vector Autoregression model from the statsmodels library.
from statsmodels.tsa.api import VAR

def calculate_summary_matrix(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Fits a VAR model and calculates a summary matrix of causal influences.

    Args:
        data (np.ndarray): Input time series data, with shape (n_samples, n_variables).
        args (argparse.Namespace): Command-line arguments for model configuration.

    Returns:
        np.ndarray: A summary matrix where M[i, j] is the causal strength from node i to node j.
    """
    # 1. Instantiate the Vector Autoregression (VAR) model with the time series data.
    model = VAR(data)

    # 2. Fit the model to the data, considering up to a maximum number of lags.
    # The result contains the coefficient matrices for each lag.
    result = model.fit(maxlags=args.max_lags)
    
    # 3. Process the raw coefficient matrices from the fitted model.
    if not result.coefs.any():
        raise RuntimeError("Model fitting did not produce any coefficient matrices.")
    
    print("Shape of raw coefficient matrices (lags, vars, vars):", np.array(result.coefs).shape)

    # Optional: Save the raw, absolute-valued coefficient matrices for each lag to a file.
    if args.output:
        try:
            # Determine the output filename for the raw results.
            output_path = args.output
            if output_path.endswith(".txt"):
                save_filename = output_path.replace(".txt", "_raw_result.npy")
            else: 
                # Create a default name if the output is not a .txt file.
                save_filename = "var_raw_results.npy"
            
            # Stack the absolute values of each lag's matrix into a single 3D array.
            abs_matrices = [np.abs(matrix) for matrix in result.coefs]
            stacked_matrices_to_save = np.stack(abs_matrices)
            
            # Save the 3D array to a .npy file for later analysis.
            np.save(save_filename, stacked_matrices_to_save)
            print(f"Saved all {len(abs_matrices)} raw matrices to '{save_filename}'")
        except Exception as e:
            print(f"Warning: Could not save raw matrices. Error: {e}", file=sys.stderr)
            
    # 4. Create a single summary graph of causal influences across all lags.
    # Stack the absolute values of the coefficient matrices into one 3D array.
    stacked_matrices = np.stack([np.abs(m) for m in result.coefs])
    
    # The convention for VAR coefficients B is that B[i, j] represents the influence of variable j on variable i.
    # To get a standard graph representation where M[i, j] is an edge i -> j, we need to transpose.
    # First, we take the maximum influence across all lags (axis=0).
    # Then, we transpose the result to get the desired i -> j convention.
    summary_matrix = np.max(stacked_matrices, axis=0).transpose()
    
    # Remove self-loops from the final summary graph.
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
    
    # First, add all variables as nodes to ensure isolated nodes are included.
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
        # If no output file is specified, print the graph to the console.
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Parses command-line arguments and initiates the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Discover causal relationships in time series data using a VAR model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Group for I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', default=None, help='Input CSV file path. Reads from stdin if not set.')
    io_group.add_argument('-o', '--output', default=None, help='Output adjacency list file path. Writes to stdout if not set.')
    io_group.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', default=None, help='Output weighted edgelist file path.')
    
    # --- Group for Model Configuration ---
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags for the VAR model.')

    # --- Group for Output Filtering ---
    output_group = parser.add_argument_group('Output Filtering')
    output_group.add_argument('--prune_threshold', type=float, default=0.01, help='Pruning threshold to remove weak causal links.')

    args = parser.parse_args()
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()