"""
Command-line interface for the PCMCIplus causal discovery algorithm.

This script uses the 'tigramite' library to analyze a time series from a CSV
file, infer a causal graph using PCMCIplus, and export the results.
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from tigramite import data_processing as pp

def calculate_summary_matrix(data: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    """
    MODIFIED: Calculate the summary matrix using the PCMCIplus causal discovery model.
    """

    # 1. Format data for tigramite
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=data.columns)
    
    # 2. Initialize the PCMCIplus model
    parcorr = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    
    # 3. Run the PCMCIplus algorithm
    output = pcmci.run_pcmciplus(tau_min=0, tau_max=args.max_lags, pc_alpha=args.sig_level)

    summary_matrix = pd.DataFrame(np.zeros([data.shape[1], data.shape[1]]), columns=data.columns,
                                  index=data.columns)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            for t in range(0, args.max_lags + 1):
                # print(i,j,t, output["graph"][i, j, t])
                if output["graph"][i, j, t] == '-->':
                    summary_matrix[data.columns[j]].loc[data.columns[i]] = 1
                elif output["graph"][i, j, t] == '<--':
                    summary_matrix[data.columns[i]].loc[data.columns[j]] = 1
            
    # print("Discovered summary matrix (Effect x Cause):")
    # print(summary_matrix)
    
    return summary_matrix.to_numpy()

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.
    """
    # Read data
    try:
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
        headers = df.columns.tolist()
        # Use raw time-series data; Sinkhorn handles scaling internally.
        # data = df.to_numpy()
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # Start timing
    start_time = time.time()
    
    # Calculate summary matrix
    summary_matrix = calculate_summary_matrix(df, args)
    
    # End timing
    computation_time = time.time() - start_time
    
    # Write computation time
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Create a directed graph from the summary matrix
    G = nx.DiGraph()
    
    # First add all nodes to ensure isolated nodes are included
    for header in headers:
        G.add_node(header)
    
    # Then add edges that meet the weight threshold
    for i in range(len(headers)):
        for j in range(len(headers)):
            if summary_matrix[i, j] > 0:
                weight = summary_matrix[i, j]
                G.add_edge(headers[i], headers[j], weight=weight)

    # Export weighted edgelist if requested
    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

    # Export the graph adjacency list
    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        # Write to stdout by default
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    # MODIFIED: Update description and arguments for PCMCIplus
    parser = argparse.ArgumentParser(description='Discover causal relationships in time series data using PCMCIplus.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path.')
    parser.add_argument('-o', '--output', help='Output adjacency list file path. If not set, writes to standard output.')
    parser.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    parser.add_argument('-w', '--weighted_edgelist', default=None, help='Output (unweighted) edgelist file path.')
    
    # PCMCIplus specific arguments
    parser.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags to test.')
    parser.add_argument('--sig_level', type=float, default=0.05, help='Significance level (alpha) for the independence tests.')
    
    args = parser.parse_args()
    process_data(args)

if __name__ == "__main__":
    main()