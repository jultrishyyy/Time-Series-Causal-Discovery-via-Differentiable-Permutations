"""
Command-line interface for cRNN-based Neural Granger Causality.

This script reads a multivariate time series from a CSV file, trains a 
causal Recurrent Neural Network (cRNN) model, and outputs the discovered 
causal graph in various formats. It allows for detailed configuration of
the model and training process via command-line arguments.
"""

# --- Standard Library and Third-Party Imports ---
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
import torch

# --- Custom Module Imports ---
# cRNN: The causal RNN model definition.
# train_model_gista: The training function utilizing the GISTA algorithm.
from crnn import cRNN, train_model_gista

def run_causal_discovery(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Orchestrates the training of the cRNN model and extracts the learned causal graph.

    Args:
        data (np.ndarray): The preprocessed time series data.
        args (argparse.Namespace): Command-line arguments containing model hyperparameters.

    Returns:
        np.ndarray: The discovered adjacency matrix representing the causal graph.
    """
    # Set random seeds for PyTorch and NumPy to ensure run-to-run reproducibility.
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Detect if a CUDA-enabled GPU is available and set the device accordingly.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert the NumPy data array to a PyTorch tensor and move it to the selected device.
    data_tensor = torch.from_numpy(data).float().to(device)
    
    # The training function expects the data to be a list of time series (batches).
    # For this script, we treat the entire dataset as a single batch item.
    X_train = [data_tensor]
    
    # Initialize the cRNN model with the number of time series and specified hyperparameters.
    num_series = data.shape[1]
    model = cRNN(
        num_series, 
        hidden=args.hidden_units,
        nonlinearity=args.nonlinearity
    ).to(device)

    # Train the model using the GISTA (Gradient-based Iterative Shrinkage-Thresholding Algorithm).
    # This algorithm is well-suited for learning sparse models by applying a group lasso penalty.
    print("Starting cRNN training with GISTA...")
    train_model_gista(
        model,
        X_train,
        context=args.max_lags,
        lr=args.learning_rate,
        lam=args.lambda_param,
        lam_ridge=args.lambda_ridge,
        max_iter=args.max_iter,
        check_every=args.check_every,
    )

    # Extract the learned Granger causality matrix from the model.
    # The model's raw output GC[i, j] = 1 signifies that variable j causes variable i.
    # We transpose it so the final matrix M[i, j] = 1 means i -> j for intuitive interpretation.
    gc_matrix = model.GC().cpu().numpy().T
    
    return gc_matrix

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        # Read the input data from a CSV file or standard input into a pandas DataFrame.
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
        headers = df.columns.tolist()
        
        # Standardize the data (z-score normalization) to have zero mean and unit variance.
        # This is a crucial preprocessing step for many neural network models.
        data = (df.to_numpy() - df.to_numpy().mean(axis=0)) / df.to_numpy().std(axis=0)
    except Exception as e:
        print(f"Error reading or processing input data: {e}", file=sys.stderr)
        sys.exit(1)

    # Time the causal discovery process.
    start_time = time.time()
    summary_matrix = run_causal_discovery(data, args)
    computation_time = time.time() - start_time

    # If a path is provided, save the total computation time.
    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    # Display the final adjacency matrix with headers for clarity.
    print("\nFinal Adjacency Matrix (M[i, j]=1 means i -> j):")
    print(pd.DataFrame(summary_matrix, index=headers, columns=headers))
    
    # Create a directed graph from the resulting adjacency matrix using NetworkX.
    G = nx.from_numpy_array(summary_matrix, create_using=nx.DiGraph)
    
    # Relabel the graph nodes from integer indices to their original variable names.
    nx.relabel_nodes(G, {i: name for i, name in enumerate(headers)}, copy=False)

    # Save the graph to a weighted edgelist file if specified.
    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

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
        description='Discover causal relationships in time series using cRNN Neural Granger Causality.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Group for Input/Output Arguments ---
    io_group = parser.add_argument_group('I/O Arguments')
    io_group.add_argument('-i', '--input', help='Input CSV file path. Reads from stdin if not set.')
    io_group.add_argument('-o', '--output', help='Output adjacency list file path. Writes to stdout if not set.')
    io_group.add_argument('-t', '--time', help='Output file path to save computation time.')
    io_group.add_argument('-w', '--weighted_edgelist', help='Output weighted edgelist file path.')
    
    # --- Group for Model and Training Hyperparameters ---
    model_group = parser.add_argument_group('Model & Training Arguments')
    model_group.add_argument('--max_lags', type=int, default=5, help='Context window length for preparing training data.')
    model_group.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the RNN.')
    model_group.add_argument('--nonlinearity', type=str, default='relu', choices=['relu', 'tanh'], help='Nonlinearity for the RNN cells.')
    model_group.add_argument('--max_iter', type=int, default=1000, help='Maximum number of training iterations.')
    model_group.add_argument('--check_every', type=int, default=100, help='Frequency of printing training progress.')
    model_group.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the GISTA optimizer.')
    model_group.add_argument('--lambda_param', type=float, default=0.1, help='Regularization strength for the group sparsity penalty (L1).')
    model_group.add_argument('--lambda_ridge', type=float, default=0.01, help='Regularization strength for the ridge penalty (L2).')
    model_group.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # Parse the provided arguments from the command line.
    args = parser.parse_args()
    
    # Start the main data processing function.
    process_data(args)

# Standard Python entry point.
if __name__ == '__main__':
    main()