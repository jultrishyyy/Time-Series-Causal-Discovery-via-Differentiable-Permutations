import os
import io
import sys
import subprocess
import numpy as np
import pandas as pd

# Define the root directory relative to this script's location
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

print("ROOT_DIR:", ROOT_DIR)


def get_metrics(label_summary_matrix, estimated_summary_matrix):
    """Calculates F1, Precision, and Recall."""
    # Ensure matrices are binary
    label_summary_matrix = (label_summary_matrix > 0).astype(int)
    estimated_summary_matrix = (estimated_summary_matrix > 0).astype(int)

    tp = np.sum((label_summary_matrix == 1) & (estimated_summary_matrix == 1))
    fp = np.sum((label_summary_matrix == 0) & (estimated_summary_matrix == 1))
    fn = np.sum((label_summary_matrix == 1) & (estimated_summary_matrix == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score, precision, recall

def reconstruct_matrix_from_adjlist(adjlist_path: str, headers: list) -> np.ndarray:
    """
    Reads a NetworkX adjacency list file and reconstructs a binary adjacency matrix.
    The output matrix follows the convention where columns are causes and rows are effects
    (i.e., matrix[i, j] = 1 represents a causal link from node j to node i).
    """

    # headers.remove('timestamp') if 'timestamp' in headers else None
    
    n = len(headers)
    name_to_idx = {name: i for i, name in enumerate(headers)}
    matrix = np.zeros((n, n), dtype=int)

    if not os.path.exists(adjlist_path):
        print(f"Warning: Adjacency list file not found at {adjlist_path}", file=sys.stderr)
        return matrix

    with open(adjlist_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            # In an adjacency list, the first item is the source (cause)
            source_node_name = parts[0]
            
            if source_node_name not in name_to_idx:
                continue
            source_idx = name_to_idx[source_node_name] # Index of the cause

            # Subsequent items are the neighbors/destinations (effects)
            for neighbor_name in parts[1:]:
                if neighbor_name in name_to_idx:
                    dest_idx = name_to_idx[neighbor_name] # Index of the effect
                    
                    # Set matrix[effect, cause] = 1 for the edge cause -> effect
                    matrix[dest_idx, source_idx] = 1            
    return matrix

def run_command(command: str, timeout: int):
    """Executes a command line command with a timeout and prints its output."""
    print(f"Executing with timeout of {timeout}s: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            print("--- SCRIPT OUTPUT ---\n" + result.stdout)
        if result.stderr:
            print("--- SCRIPT ERROR ---\n" + result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"!!! COMMAND FAILED WITH EXIT CODE {result.returncode} !!!", file=sys.stderr)
        print("-" * 20)
        return True # Indicates success
    except subprocess.TimeoutExpired:
        print(f"!!! TIMEOUT ERROR: Command exceeded {timeout} seconds and was terminated. !!!", file=sys.stderr)
        print("-" * 20)
        return False # Indicates failure due to timeout

def evaluate_and_get_metrics(command: str, output_adj_path: str, label_matrix: np.ndarray, headers: list, timeout: int):
    """Runs a command with a timeout, reconstructs the matrix, and returns metrics."""
    success = run_command(command, timeout=timeout)
    if not success:
        return -1, -1, -1 # Return NaN on timeout

    pred_matrix = reconstruct_matrix_from_adjlist(output_adj_path, headers)
    f1, p, r = get_metrics(label_matrix, pred_matrix)
    print(f"📊 Evaluation: F1={f1:.3f}, Precision={p:.3f}, Recall={r:.3f}\n")
    return f1, p, r

def run_sinkhorn(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the Sinkhorn method in unsupervised mode."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "sinkhorn", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"sinkhorn_adj_lag{lag}.txt")
    unsupervised_model_path = os.path.join(path_dict['output_path'], f"sinkhorn_model_lag{lag}.pth")
    output_weight_path = os.path.join(path_dict['output_path'], f"sinkhorn_weight_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --random_seed 42 "
        f"--input {path_dict['data_filename']} --output {output_adj_path} --save_path {unsupervised_model_path} "
        # f"--weighted_edgelist {output_weight_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'sinkhorn_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_varlingam(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the VARLiNGAM method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "varlingam", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"varlingam_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} "
        f"--input {path_dict['data_filename']} --output {output_adj_path} "
        f"--prune True "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'varlingam_time_lag{lag}.txt')} "
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_VAR(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the VAR method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "VAR", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"VAR_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'VAR_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_TCDF(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the TCDF method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "TCDF", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"TCDF_adj_lag{lag}.txt")
    
    # Note: TCDF script might not use max_lags, but we keep the parameter for consistency
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--time {os.path.join(path_dict['output_path'], f'TCDF_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_Dynotears(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the Dynotears method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "Dynotears", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"Dynotears_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'Dynotears_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_mvgc(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the MVGC method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "mvgc", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"mvgc_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'mvgc_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_pcmici(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the PCMCI method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "pcmci", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"pcmci_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'pcmci_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_pcgce(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the PCGCE method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "pcgce", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"pcgce_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'pcgce_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_timinio(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the TiMINo method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "timino", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"timino_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'timino_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_tsFCI(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the tsFCI method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "tsFCI", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"tsFCI_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'tsFCI_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_cLSTM(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the cLSTM method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "cLSTM", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"cLSTM_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'cLSTM_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_cMLP(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the cMLP method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "cMLP", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"cMLP_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'cMLP_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def run_cRNN(path_dict, headers, true_matrix, display_name, timeout, lag):
    """Runs the cRNN method."""
    discovery_script_path = os.path.join(ROOT_DIR, "algorithms", "cRNN", "csvif.py")
    output_adj_path = os.path.join(path_dict['output_path'], f"cRNN_adj_lag{lag}.txt")
    
    command = (
        f"python {discovery_script_path} --input {path_dict['data_filename']} "
        f"--output {output_adj_path} "
        f"--max_lags {lag} "
        f"--time {os.path.join(path_dict['output_path'], f'cRNN_time_lag{lag}.txt')}"
    )
    return evaluate_and_get_metrics(command, output_adj_path, true_matrix, headers, timeout)

def main():
    TIMEOUT_SECONDS = 10800  # 3 hours

    paths = [
        ['Middleware_oriented_message_Activity', '/monitoring_metrics_1.csv', 'MoM_1/', 'MoM 1'],
        ['Middleware_oriented_message_Activity', '/monitoring_metrics_2.csv', 'MoM_2/', 'MoM 2'],
        ['Storm_Ingestion_Activity', '/storm_data_normal.csv', '', 'Ingestion'],
        ['Web_Activity', '/preprocessed_1.csv', 'Web_1/', 'Web 1'],
        ['Web_Activity', '/preprocessed_2.csv', 'Web_2/', 'Web 2'],
        ['Antivirus_Activity', '/preprocessed_1.csv', 'Antivirus_1/', 'Antivirus 1'],
        ['Antivirus_Activity', '/preprocessed_2.csv', 'Antivirus_2/', 'Antivirus 2'],
        # ['SWaT', '/preprocessed_2.csv', '', 'SWaT'],
        # ['Flood', '/rivers_ts_flood_preprocessed.csv', '', 'Flood'],
        # ['Bavaria', '/rivers_ts_bavaria_preprocessed.csv', '', 'Bavaria'],
        # ['East_germany', '/rivers_ts_east_germany_preprocessed.csv', '', 'East germany'],
    ]
    
    methods_to_run = {
        # 'VAR': run_VAR,
        # 'MVGC': run_mvgc,
        # 'cLSTM': run_cLSTM,
        # 'cMLP': run_cMLP,
        # 'cRNN': run_cRNN,
        # 'TCDF': run_TCDF,  
        # 'PCMCI+': run_pcmici,
        # 'tsFCI': run_tsFCI,
        # 'PCGCE': run_pcgce,
        # 'DYNOTEARS': run_Dynotears,
        # 'VARLiNGAM': run_varlingam,
        # 'TiMINO': run_timinio,
        'sinkhorn': run_sinkhorn,
    }
    
    # lags_to_test = [3, 5, 10, 15]
    lags_to_test = [3]

    for lag in lags_to_test:
        print("\n" + "#"*80)
        print(f"## STARTING EXPERIMENTS FOR LAG = {lag} ##")
        print("#"*80)

        # --- Initialize DataFrames for the current lag ---
        column_names = [p[3] for p in paths]
        index_names = list(methods_to_run.keys())
        
        f1_df = pd.DataFrame(index=index_names, columns=column_names, dtype=float)
        recall_df = pd.DataFrame(index=index_names, columns=column_names, dtype=float)
        precision_df = pd.DataFrame(index=index_names, columns=column_names, dtype=float)

        for dataset_folder, data_file, subfolder, display_name in paths:
            print("\n" + "="*80)
            print(f"🚀 PROCESSING DATASET: {display_name} (Lag={lag})")
            print("="*80)

            # --- Modified output_path to include the current lag ---
            path_dict = {
                'data_filename': os.path.join(ROOT_DIR, "data", dataset_folder, data_file.lstrip('/')),
                'label_filename': os.path.join(ROOT_DIR, "data", dataset_folder, 'summary_matrix.npy'),
                'partial_gt_path': os.path.join(ROOT_DIR, "data", dataset_folder, 'partial_summary_matrix.npy'),
                'output_path': os.path.join(ROOT_DIR, "result", dataset_folder, subfolder, str(lag)) if subfolder else os.path.join(ROOT_DIR, "result", dataset_folder, str(lag)),
            }
            os.makedirs(path_dict['output_path'], exist_ok=True)

            try:
                headers = pd.read_csv(path_dict['data_filename'], nrows=0).columns.tolist()
                true_matrix = np.load(path_dict['label_filename'])
            except FileNotFoundError as e:
                print(f"Error loading data/label file: {e}. Skipping dataset.", file=sys.stderr)
                continue
                
            for method_name, method_func in methods_to_run.items():
                print(f"\n--- Running Method: {method_name} on {display_name} ---")
                # --- Pass the current lag to the method function ---
                f1, p, r = method_func(path_dict, headers, true_matrix, display_name, timeout=TIMEOUT_SECONDS, lag=lag)
                
                f1_df.loc[method_name, display_name] = f1
                recall_df.loc[method_name, display_name] = r
                precision_df.loc[method_name, display_name] = p

            # --- Save Final Results to a CSV file specific to the current lag ---
            # output_csv_path = os.path.join(ROOT_DIR, "result", f"final_results_lag{lag}.csv")
            output_csv_path = os.path.join(ROOT_DIR, "result", f"test.csv")
            try:
                with open(output_csv_path, 'w', newline='') as f:
                    f.write(f"Results for Lag = {lag}\n")
                    f.write("\nF1\n")
                    f1_df.dropna(how='all', axis=0).to_csv(f, float_format='%.4f')
                    f.write("\nrecall\n")
                    recall_df.dropna(how='all', axis=0).to_csv(f, float_format='%.4f')
                    f.write("\nprecision\n")
                    precision_df.dropna(how='all', axis=0).to_csv(f, float_format='%.4f')
                print("\n" + "="*80)
                print(f"✅ FINAL RESULTS FOR LAG={lag} SAVED TO: {output_csv_path}")
                print("="*80)
            except Exception as e:
                print(f"Error saving final CSV results for lag {lag}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()