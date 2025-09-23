"""
Command-line interface for the TiMINo causal discovery algorithm.

This script reads a time series from a CSV file, applies the TiMINo algorithm
to infer a causal graph (DAG), and exports the resulting structure.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time

# --- Custom Module Imports ---
# timino_dag: The core function from the 'timino' library that runs the algorithm.
from timino import timino_dag 

def run_causal_discovery(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    A wrapper function that calls the TiMINo DAG discovery algorithm.

    Args:
        data (np.ndarray): The input time series data as a NumPy array.
        args (argparse.Namespace): Command-line arguments containing TiMINo parameters.

    Returns:
        np.ndarray: The discovered summary adjacency matrix.
    """
    print("Starting TiMINo DAG discovery...")
    
    # Call the main TiMINo function with data and specified hyperparameters.
    # The returned matrix format is M[i, j] = 1, which represents an edge i -> j.
    summary_matrix = timino_dag(
        M=data,
        alpha=args.alpha,
        max_lag=args.max_lags,
        instant=args.instant,
    )
    print("TiMINo DAG discovery finished.")
    
    # Ensure there are no self-loops in the final graph.
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
        # Convert the pandas DataFrame to a NumPy array for the algorithm.
        data = df.to_numpy()
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Run Causal Discovery ---
    start_time = time.time()
    summary_matrix = run_causal_discovery(data, args)
    computation_time = time.time() - start_time

    # --- Save Computation Time ---
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Display the final matrix for user verification.
    print("\nFinal Adjacency Matrix (M[i, j]=1 means i -> j):")
    print(pd.DataFrame(summary_matrix, index=headers, columns=headers))
    
    # --- Create and Save the Output Graph ---
    
    # 1. Initialize an empty directed graph.
    G = nx.DiGraph()
    
    # 2. Add all variables as nodes to ensure isolated nodes are included.
    G.add_nodes_from(headers)
    
    # 3. Add edges based on the non-zero entries in the summary matrix.
    # Get the row and column indices where a causal link exists.
    rows, cols = np.where(summary_matrix == 1)
    # Create an edge for each found link. The convention M[i, j] means i -> j.
    for i, j in zip(rows, cols):
        G.add_edge(headers[i], headers[j], weight=1.0)

    # --- Export Results ---
    if args.weighted_edgelist:
        # TiMINo's output is binary, so weights are implicitly 1.0.
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        # If no output file is specified, print the graph to the console.
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Parses command-line arguments and initiates the data processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Discover causal relationships in time series data using the TiMINo algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Group for I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', help='Input CSV file path. Reads from stdin if not set.')
    io_group.add_argument('-o', '--output', help='Output adjacency list file path. Writes to stdout if not set.')
    io_group.add_argument('-t', '--time', help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', help='Output weighted edgelist file path.')
    
    # --- Group for TiMINo Specific Arguments ---
    model_group = parser.add_argument_group('TiMINo Specific Arguments')
    model_group.add_argument('--max_lags', type=int, default=3, help='Maximum number of time lags to consider.')
    model_group.add_argument('--alpha', type=float, default=0.05, help='Significance level for independence tests.')
    model_group.add_argument('--instant', type=int, default=1, help='Lag for instantaneous effects (0 means no instantaneous effects).')

    args = parser.parse_args()
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()