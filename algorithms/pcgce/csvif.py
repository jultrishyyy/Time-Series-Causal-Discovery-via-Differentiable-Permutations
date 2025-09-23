"""
Command-line interface for the PCGCE causal discovery algorithm.

This script reads time series data and uses the PC-based Granger Causal 
Inference (PCGCE) algorithm to infer a causal summary graph.
"""

import argparse
import pandas as pd
import networkx as nx
import sys
import time

# Import the main class from our logic file
from pcgce import PCGCE

def calculate_summary_matrix(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Wrapper function to call the PCGCE analysis and return the summary adjacency matrix.
    """
    print(f"Running PCGCE analysis with max_lags={args.max_lags} and sig_level={args.sig_level}...")
    
    # 1. Instantiate and fit the PCGCE model
    model = PCGCE(
        series=data,
        max_lags=args.max_lags,
        sig_level=args.sig_level,
    )
    summary_graph = model.fit()
    
    # 2. Convert the NetworkX graph output to a pandas adjacency matrix
    # The graph has i -> j for a causal link, which matches the desired matrix format.
    adj_matrix = nx.to_pandas_adjacency(summary_graph, nodelist=data.columns.tolist(), dtype=int)
    
    return adj_matrix

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.
    """
    # --- Main Processing Pipeline ---
    try:
        data = pd.read_csv(args.input, header=0)
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
    # Transpose the matrix to get the standard i->j representation for graph creation
    G = nx.from_pandas_adjacency(summary_matrix.transpose(), create_using=nx.DiGraph)
    
    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)
        print(f"Saved weighted edgelist to: {args.weighted_edgelist}")

    if args.output:
        nx.write_adjlist(G, args.output)
        print(f"Saved adjacency list to: {args.output}")
    else:
        print("\n--- Discovered Causal Graph (Adjacency List) ---")
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """Main function to parse arguments and run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Discover causal summary graphs from time series data using PCGCE.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- I/O Arguments ---
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path.')
    parser.add_argument('-o', '--output', default=None, help='Output adjacency list file path. Writes to stdout if not set.')
    parser.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    parser.add_argument('-w', '--weighted_edgelist', default=None, help='Output weighted edgelist file path (weights will be 1).')

    # --- PCGCE Specific Arguments ---
    parser.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to summarize into the "past" variable.')
    parser.add_argument('--sig_level', type=float, default=0.05, help='Significance level (alpha) for the conditional independence tests.')
    
    args = parser.parse_args()

    process_data(args)

if __name__ == '__main__':
    main()