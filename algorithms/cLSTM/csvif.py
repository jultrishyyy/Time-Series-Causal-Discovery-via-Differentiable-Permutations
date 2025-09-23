"""
Command-line interface for Neural Granger Causality.

Reads a CSV file, trains a cLSTM model, and outputs the discovered causal graph.
"""
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
import torch

from clstm import cLSTM, train_model_gista

def run_causal_discovery(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    Trains the cLSTM model and extracts the Granger causality matrix.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data preprocessing
    data_tensor = torch.from_numpy(data).float().to(device)
    
    # The training function expects a list of time series (batches)
    # Here we treat the whole dataset as a single batch item
    X_train = [data_tensor]
    
    # Initialize model
    num_series = data.shape[1]
    model = cLSTM(num_series, hidden=args.hidden_units).to(device)

    # Train model
    train_model_gista(
        model,
        X_train,
        context=args.max_lags,
        lr=args.learning_rate,
        lam=args.lambda_param,
        lam_ridge=args.lambda_ridge,
        max_iter=args.max_iter,
    )

    # Extract the Granger causality matrix
    # The result GC[i, j] = 1 means j -> i, so we transpose it
    gc_matrix = model.GC().cpu().numpy().T
    
    return gc_matrix

def process_data(args: argparse.Namespace):
    """
    Main processing pipeline: reads data, runs causal discovery, and saves results.
    """
    try:
        df = pd.read_csv(args.input if args.input else sys.stdin, header=0)
        headers = df.columns.tolist()
        # Normalize data to have zero mean and unit variance
        data = (df.to_numpy() - df.to_numpy().mean(axis=0)) / df.to_numpy().std(axis=0)
    except Exception as e:
        print(f"Error reading or processing input data: {e}", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    summary_matrix = run_causal_discovery(data, args)
    computation_time = time.time() - start_time

    if args.time:
        with open(args.time, 'w') as f:
            f.write(f"{computation_time:.6f}\n")
    
    print("\nFinal Adjacency Matrix (M[i, j]=1 means i -> j):")
    print(pd.DataFrame(summary_matrix, index=headers, columns=headers))
    
    G = nx.from_numpy_array(summary_matrix, create_using=nx.DiGraph)
    nx.relabel_nodes(G, {i: name for i, name in enumerate(headers)}, copy=False)

    if args.weighted_edgelist:
        nx.write_weighted_edgelist(G, args.weighted_edgelist)

    if args.output:
        nx.write_adjlist(G, args.output)
    else:
        for line in nx.generate_adjlist(G):
            print(line)

def main():
    parser = argparse.ArgumentParser(
        description='Discover causal relationships in time series using Neural Granger Causality.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- I/O Arguments ---
    parser.add_argument('-i', '--input', help='Input CSV file path. Reads from stdin if not set.')
    parser.add_argument('-o', '--output', help='Output adjacency list file path. Writes to stdout if not set.')
    parser.add_argument('-t', '--time', help='Output file path to save computation time.')
    parser.add_argument('-w', '--weighted_edgelist', help='Output weighted edgelist file path.')
    
    # --- Model & Training Arguments ---
    parser.add_argument('--max_lags', type=int, default=5, help='Context window length (autoregressive order).')
    parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the LSTM.')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of training iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--lambda_param', type=float, default=0.1, help='Lambda for the group sparsity penalty.')
    parser.add_argument('--lambda_ridge', type=float, default=0.01, help='Lambda for the ridge penalty.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    

    args = parser.parse_args()
    process_data(args)

if __name__ == '__main__':
    main()