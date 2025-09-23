"""
Command-line interface for Multivariate Granger Causality (MVGC) analysis.

This script acts as a user-friendly wrapper for an MVGC implementation. It reads
a multivariate time series from a CSV file, performs the Granger causality test
to identify causal links, and outputs the resulting causal network in various
standard graph formats.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
from pathlib import Path

# Import the main function from our logic file
from mvgc import MVGC

def calculate_summary_matrix(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Wrapper function to call the MVGC analysis and return the resulting matrix.
    """
    print(f"Running MVGC analysis with max_lags={args.max_lags} and sig_level={args.sig_level}...")
    
    # The returned matrix A[i, j] means j -> i.
    # We transpose it so that A[i, j] means i -> j, which is standard for graph libraries.
    causal_matrix = MVGC(
        data=data,
        max_lags=args.max_lags,
        sig_level=args.sig_level,
    ).transpose()
    
    return causal_matrix

def main():
    """Main function to parse arguments and run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Discover causal relationships in time series data using Multivariate Granger Causality.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- I/O Arguments ---
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path.')
    parser.add_argument('-o', '--output', default=None, help='Output adjacency list file path. Writes to stdout if not set.')
    parser.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    parser.add_argument('-w', '--weighted_edgelist', default=None, help='Output weighted edgelist file path (weights will be 1).')

    # --- MVGC Configuration ---
    parser.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to consider for the VAR model.')
    parser.add_argument('--sig_level', type=float, default=0.05, help='Significance level (alpha) for the Granger causality F-test.')

    args = parser.parse_args()

    # --- Main Processing Pipeline ---
    try:
        data = pd.read_csv(args.input, header=0)
        headers = data.columns.tolist()
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    
    summary_matrix = calculate_summary_matrix(data, args)
    
    computation_time = time.time() - start_time
    print(f"\nComputation finished in {computation_time:.2f} seconds.")

    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")

    # --- Create and Save the Output Graph ---
    G = nx.from_pandas_adjacency(summary_matrix, create_using=nx.DiGraph)
    
    if args.weighted_edgelist:
        # Since the matrix is 0/1, weights will be 1.0 for existing edges
        nx.write_weighted_edgelist(G, args.weighted_edgelist)
        print(f"Saved weighted edgelist to: {args.weighted_edgelist}")

    if args.output:
        nx.write_adjlist(G, args.output)
        print(f"Saved adjacency list to: {args.output}")
    else:
        print("\n--- Discovered Causal Graph (Adjacency List) ---")
        for line in nx.generate_adjlist(G):
            print(line)

if __name__ == '__main__':
    main()