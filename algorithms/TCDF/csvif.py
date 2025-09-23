"""
Command-line interface for the Temporal Causal Discovery Framework (TCDF).

This script provides a user-friendly way to run the TCDF algorithm on a time
series dataset from a CSV file and export the discovered causal graph.
"""

import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import sys
import time
import os

# --- Path Correction for TCDF Library ---
# This block is necessary because the TCDF library is structured in a way
# that requires its parent directory to be in Python's path to find modules.
try:
    # Get the directory where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assume the TCDF library is in a subdirectory named 'TCDF'.
    tcdf_lib_path = os.path.join(script_dir, 'TCDF')

    # Add the TCDF library path to the start of Python's search path if not already present.
    if tcdf_lib_path not in sys.path:
        sys.path.insert(0, tcdf_lib_path)
finally:
    # This import will now succeed because the path is correctly set.
    import TCDF

def run_tcdf_and_get_causes(datafile: str, args: argparse.Namespace):
    """
    Runs the core TCDF discovery process for each variable in the dataset.

    This function iterates through each time series, treating it as a potential
    effect, and calls the main TCDF function to find its causes.

    Args:
        datafile (str): Path to the input CSV data file.
        args (argparse.Namespace): Command-line arguments with TCDF hyperparameters.

    Returns:
        tuple[dict, list]: A dictionary of discovered causal links and a list of variable names.
    """
    df_data = pd.read_csv(datafile, header=0)
    allcauses = {}
    columns = list(df_data.columns)

    # Loop through each column, treating it as the target (effect) variable.
    for c in columns:
        # Get the integer index of the current target column.
        idx = df_data.columns.get_loc(c)
        
        # Call the main TCDF function to find causes for the target 'c'.
        # We only need the first return value, which is the list of discovered causes.
        causes, _, _, _ = TCDF.findcauses(
            c,
            cuda=args.cuda,
            epochs=args.epochs,
            kernel_size=args.kernel_size,
            layers=args.hidden_layers + 1,
            log_interval=args.log_interval,
            lr=args.learning_rate,
            optimizername=args.optimizer,
            seed=args.seed,
            dilation_c=args.dilation_coefficient,
            significance=args.significance,
            file=datafile
        )
        # Store the results. The dictionary format is: {effect_index: [cause_index_1, cause_index_2, ...]}
        allcauses[idx] = causes
    print(f"Discovered causes (effect_idx: [cause_indices]): {allcauses}")

    return allcauses, columns

def calculate_summary_matrix(data_path: str, args: argparse.Namespace):
    """
    Calculates the summary adjacency matrix from the TCDF output.

    Args:
        data_path (str): Path to the input CSV data file.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        tuple[np.ndarray, list]: The computed summary matrix and the list of variable names.
    """
    # 1. Run the TCDF discovery process to get the dictionary of causal links.
    allcauses, headers = run_tcdf_and_get_causes(data_path, args)

    # 2. Convert the dictionary output into a standard NumPy adjacency matrix.
    n_features = len(headers)
    summary_matrix = np.zeros((n_features, n_features), dtype=int)

    # Iterate through the dictionary of discovered causes.
    for effect_idx, cause_indices in allcauses.items():
        for cause_idx in cause_indices:
            # Set matrix[row, col] = 1. Our convention here is matrix[effect, cause].
            # This means a '1' at M[i, j] signifies a causal link from j -> i.
            summary_matrix[effect_idx, cause_idx] = 1
            
    # Remove any self-loops that might have been discovered.
    np.fill_diagonal(summary_matrix, 0)
    
    return summary_matrix, headers

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: runs causal discovery and saves results.
    """
    # --- Run Causal Discovery ---
    start_time = time.time()
    summary_matrix, headers = calculate_summary_matrix(args.input, args)
    computation_time = time.time() - start_time

    # --- Save Computation Time ---
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # --- Create and Save the Output Graph ---
    
    # Create a directed graph from the summary matrix.
    # The matrix was created with the convention M[effect, cause] = 1 (cause -> effect).
    # NetworkX's from_numpy_array expects M[cause, effect] = 1 for an edge cause -> effect.
    # Therefore, we must transpose the matrix to match the NetworkX convention.
    G = nx.from_numpy_array(summary_matrix.T, create_using=nx.DiGraph)
    
    # Relabel the integer-indexed nodes with their original variable names.
    nx.relabel_nodes(G, {i: h for i, h in enumerate(headers)}, copy=False)

    # --- Export Results ---
    if args.weighted_edgelist:
        # TCDF output is binary (link exists or not), so weights are implicitly 1.0.
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        # If no output file is given, print the adjacency list to the console.
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Parses command-line arguments and initiates the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Discover causal relationships in time series data using TCDF.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Group for I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', required=True, help='Input CSV file path.')
    io_group.add_argument('-o', '--output', default=None, help='Output adjacency list file path. Writes to stdout if not set.')
    io_group.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', default=None, help='Output weighted edgelist file path.')
    
    # --- Group for TCDF Specific Hyperparameters ---
    model_group = parser.add_argument_group('TCDF Specific Arguments')
    model_group.add_argument('--cuda', action="store_true", default=False, help='Enable CUDA for GPU acceleration.')
    model_group.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    model_group.add_argument('--kernel_size', type=int, default=5, help='Size of the convolutional kernel (window size).')
    model_group.add_argument('--hidden_layers', type=int, default=0, help='Number of hidden layers in the depthwise convolution.')
    model_group.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    model_group.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop'], help='Optimizer to use.')
    model_group.add_argument('--log_interval', type=int, default=500, help='Epoch interval to report training loss.')
    model_group.add_argument('--seed', type=int, default=1111, help='Random seed for reproducibility.')
    model_group.add_argument('--dilation_coefficient', type=int, default=4, help='Dilation coefficient for temporal convolutions.')
    model_group.add_argument('--significance', type=float, default=0.8, help="Significance threshold for validating a cause.")

    args = parser.parse_args()
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()