"""
Command-line interface for the tsFCI (Time Series Fast Causal Inference) algorithm.

This script runs the tsFCI algorithm on time series data from a CSV file,
summarizes the resulting time-stamped graph, and exports the causal structure.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time

# --- Custom Module Imports ---
# tsfci: The main function that runs the time series FCI algorithm.
# get_time_slice: A helper function to parse time indices from variable names.
from tsfci import tsfci, get_time_slice

def create_summary_matrix(adj: dict, var_names: list[str]) -> np.ndarray:
    """
    Creates a summary matrix from the detailed time-stamped PAG.

    The tsFCI algorithm outputs a graph over variables at different time lags 
    (e.g., VarA_t-1, VarB_t). This function simplifies that complex output into
    a single matrix representing causal links between the original variables.
    
    Args:
        adj (dict): The adjacency list of the PAG from tsFCI.
        var_names (list[str]): The list of original variable names.

    Returns:
        np.ndarray: A summary adjacency matrix where M[i, j] = 1 means i -> j.
    """
    num_vars = len(var_names)
    summary_matrix = np.zeros((num_vars, num_vars))
    
    # Create a mapping from a base variable name (e.g., 'VarA') to its column index.
    name_to_idx = {name: i for i, name in enumerate(var_names)}

    # Iterate through each node and its neighbors in the time-stamped graph.
    for u, neighbors in adj.items():
        # Parse the base name and time slice from the 'from' node (e.g., 'VarA_t-1').
        time_u = get_time_slice(u)
        base_u_name = u.split('_t')[0]
        
        # Skip if the variable is not one of the original ones.
        if base_u_name not in name_to_idx: continue
        u_idx = name_to_idx[base_u_name]

        # Iterate through the neighbors of the current node.
        for v in neighbors:
            # Parse the base name and time slice from the 'to' node (e.g., 'VarB_t').
            time_v = get_time_slice(v)
            base_v_name = v.split('_t')[0]

            if base_v_name not in name_to_idx: continue
            v_idx = name_to_idx[base_v_name]

            # The core simplification logic: if an edge exists from the past to the present/future,
            # we summarize this as a directed causal link from the base variable u to v.
            if time_u < time_v:
                summary_matrix[u_idx, v_idx] = 1
    
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
    
    print("Starting tsFCI discovery...")
    # Run the main tsFCI algorithm. It returns a detailed Partial Ancestral Graph (PAG)
    # over time-stamped variables, represented as an adjacency dictionary.
    adj, sepsets = tsfci(
        data=data,
        var_names=headers,
        max_lag=args.max_lags,
        alpha=args.alpha,
    )
    # Convert the detailed, time-stamped PAG into a simple summary graph between base variables.
    summary_matrix = create_summary_matrix(adj, headers)
    print("tsFCI discovery finished.")

    computation_time = time.time() - start_time

    # --- Save Computation Time ---
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Display the final matrix for user verification.
    print("\nFinal Summary Matrix (M[i, j]=1 means i -> j):")
    print(pd.DataFrame(summary_matrix, index=headers, columns=headers))
    
    # --- Create and Save the Output Graph ---
    # Create a directed graph directly from the summary matrix.
    # The matrix was constructed so that M[i, j] correctly represents an edge i -> j.
    G = nx.from_numpy_array(summary_matrix, create_using=nx.DiGraph)
    nx.relabel_nodes(G, {i: name for i, name in enumerate(headers)}, copy=False)

    # --- Export Results ---
    if args.weighted_edgelist:
        # The summary graph is binary, so edge weights will be 1.0.
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
        description='Discover causal relationships in time series data using the tsFCI algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Group for I/O Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', help='Input CSV file path. Reads from stdin if not set.')
    io_group.add_argument('-o', '--output', help='Output adjacency list file path. Writes to stdout if not set.')
    io_group.add_argument('-t', '--time', help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', help='Output weighted edgelist file path.')
    
    # --- Group for tsFCI Specific Arguments ---
    model_group = parser.add_argument_group('tsFCI Specific Arguments')
    model_group.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to consider.')
    model_group.add_argument('--alpha', type=float, default=0.05, help='Significance level for conditional independence tests.')
    
    args = parser.parse_args()
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()