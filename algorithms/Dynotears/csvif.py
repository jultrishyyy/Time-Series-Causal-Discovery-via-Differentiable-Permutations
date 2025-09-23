"""
Command-line interface for time series causal discovery using DYN-NOTEARS.

This script leverages the `causalnex` library to analyze a multivariate time 
series from a CSV file. It applies the DYN-NOTEARS algorithm to learn a 
causal graph, which is then outputted in various formats.
"""

# --- Standard Library and Third-Party Imports ---
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time

# --- Custom Module Imports ---
# from_pandas_dynamic: The CausalNex implementation of the DYN-NOTEARS algorithm.
from causalnex.structure.dynotears import from_pandas_dynamic


def calculate_summary_matrix(df: pd.DataFrame, args: argparse.Namespace):
    """
    Calculates the causal summary matrix using the DYN-NOTEARS algorithm.

    Args:
        df (pd.DataFrame): The input time series data, where columns are variables.
        args (argparse.Namespace): Command-line arguments containing algorithm parameters.

    Returns:
        tuple[np.ndarray, list]: A tuple containing the computed summary matrix 
                                 and the list of feature names.
    """
    feature_names = df.columns.tolist()
    print("Feature names:", feature_names)
    n_features = len(feature_names)

    # 1. Run the DYN-NOTEARS algorithm.
    # 'p' is the hyperparameter for the maximum lag to consider.
    # 'w_threshold' filters out weak causal links below this value.
    # The function returns a StructureModel, which is a graph-like object.
    sm = from_pandas_dynamic(df, p=args.max_lags, w_threshold=args.w_threshold)

    # DYN-NOTEARS creates nodes with names like 'FeatureA_lag1'.
    # We need to map these lagged names back to their original feature names.
    tname_to_name_dict = dict()
    for tname in sm.nodes:
        # Split 'FeatureA_lag1' into ('FeatureA', '1') and take the first part.
        base_name = tname.split("_lag")[0]
        tname_to_name_dict[tname] = base_name

    # 2. Process the results into a standard adjacency matrix.
    # Initialize an empty matrix with original feature names for rows and columns.
    summary_matrix = pd.DataFrame(np.zeros([n_features, n_features]), 
                                  columns=df.columns,
                                  index=df.columns)
                                  
    # Iterate through all discovered causal edges in the StructureModel.
    for cause_node, effect_node in sm.edges:
        # Map the lagged cause and effect node names back to their base feature names.
        cause_name = tname_to_name_dict[cause_node]
        effect_name = tname_to_name_dict[effect_node]
        
        # Set the corresponding cell in the matrix to 1 to indicate a causal link.
        # The format summary_matrix[column].loc[row] means row -> column.
        # Here, it's summary_matrix[effect].loc[cause] = 1, meaning cause -> effect.
        summary_matrix[effect_name].loc[cause_name] = 1

    # Convert the pandas DataFrame to a NumPy array for further processing.
    summary_matrix = summary_matrix.to_numpy()

    return summary_matrix, feature_names

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        # Read the input data from a CSV file into a pandas DataFrame.
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # Time the causal discovery process.
    start_time = time.time()
    summary_matrix, headers = calculate_summary_matrix(df, args)
    computation_time = time.time() - start_time
    
    # Remove self-loops (e.g., A -> A) by setting the diagonal of the matrix to zero.
    np.fill_diagonal(summary_matrix, 0)
    
    print("Final Adjacency Matrix (M[i, j]=1 means i -> j):")
    print(pd.DataFrame(summary_matrix, index=headers, columns=headers))

    # If a path is provided, save the total computation time.
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Create a NetworkX directed graph from the summary matrix.
    # The matrix format M[i, j] = 1 represents an edge from node i to node j,
    # which is the default interpretation for nx.from_numpy_array.
    G = nx.from_numpy_array(summary_matrix, create_using=nx.DiGraph)
    
    # Relabel the graph nodes from integer indices to their original variable names.
    nx.relabel_nodes(G, {i: h for i, h in enumerate(headers)}, copy=False)

    # Save the graph to an edgelist file if specified.
    if args.weighted_edgelist:
        # DYN-NOTEARS output is a binary (unweighted) graph after thresholding.
        with open(args.weighted_edgelist, 'wb') as f:
            nx.write_edgelist(G, f, data=False)

    # Save the graph to an adjacency list file or print to standard output.
    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        # If no output file is specified, print the adjacency list to the console.
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    """
    Parses command-line arguments and initiates the data processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Discover causal relationships in time series data using DYN-NOTEARS.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Group for Input/Output Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', required=True, help='Input CSV file path.')
    io_group.add_argument('-o', '--output', default=None, help='Output adjacency list file path. If not set, writes to stdout.')
    io_group.add_argument('-t', '--time', default=None, help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', default=None, help='Output (unweighted) edgelist file path.')
    
    # --- Group for DYN-NOTEARS Specific Arguments ---
    model_group = parser.add_argument_group('DYN-NOTEARS Specific Arguments')
    model_group.add_argument('--max_lags', type=int, default=5, help='Maximum number of time lags (p) to consider.')
    model_group.add_argument('--w_threshold', type=float, default=0.01, help='Threshold for edge weights to be included in the final graph.')

    # Parse the provided arguments from the command line.
    args = parser.parse_args()
    
    # Start the main data processing function.
    process_data(args)

# Standard Python entry point to execute the script.
if __name__ == '__main__':
    main()