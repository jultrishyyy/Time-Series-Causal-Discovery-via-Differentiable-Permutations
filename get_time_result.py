import os
import pandas as pd

def find_dataset_paths(root_dir):
    """
    Finds all leaf directories that contain the lag sub-folders.
    """
    dataset_paths = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if '3' in dirnames and '5' in dirnames:
            dataset_paths.append(dirpath)
    return sorted(dataset_paths)

def process_running_times():
    """
    Main function to traverse result directories, extract running times,
    and save them into four transposed CSV files, one for each time lag.
    """
    # --- Configuration ---
    result_root_dir = 'result'
    lags = [3, 5, 10, 15]

    # Map desired CSV column names to the file prefixes.
    method_map = {
        'VAR': 'VAR', 'mvgc': 'mvgc', 'cLSTM': 'cLSTM', 'cMLP': 'cMLP',
        'cRNN': 'cRNN', 'TCDF': 'TCDF', 'pcmci': 'pcmci', 'tsFCI': 'tsFCI',
        'pcgce': 'pcgce', 'Dynotears': 'Dynotears', 'VARLiNGAM': 'varlingam',
        'TiMiNo': 'timino', 'our_method': 'sinkhorn_unsupervised'
    }

    # IMPORTANT: Map your FOLDER names to the desired CSV COLUMN names.
    # Please verify the keys on the left match your folder names exactly.
    dataset_map = {
        # --- YOU MAY NEED TO EDIT THE KEYS (left side) BELOW ---
        'MoM_1': 'MoM_1',
        'MoM_2': 'MoM_2',
        'Storm_Ingestion_Activity': 'Storm_Ingestion_Activity',
        'Web_1': 'Web_1',
        'Web_2': 'Web_2',
        'Antivirus_1': 'Antivirus_1',
        'Antivirus_2': 'Antivirus_2',
        'SWaT': 'SWaT',
        'Flood': 'Flood',
        'Bavaria': 'Bavaira', # Using your spelling
        'East_germany': 'East_germany',
    }

    # Define the final order of columns in the CSV file.
    dataset_order = [
        'MoM_1', 'MoM_2', 'Storm_Ingestion_Activity', 'Web_1', 'Web_2',
        'Antivirus_1', 'Antivirus_2', 'SWaT', 'Flood', 'Bavaira', 'East_germany'
    ]

    print(f"Starting to process results from: '{result_root_dir}'")
    
    # --- Main Processing Loop ---
    for lag in lags:
        print(f"\nProcessing lag = {lag}...")
        
        # This dictionary will store data in the shape: {method: {dataset: time}}
        lag_data = {method: {} for method in method_map.keys()}
        
        dataset_paths = find_dataset_paths(result_root_dir)

        for dataset_path in dataset_paths:
            dataset_folder_name = os.path.basename(dataset_path)
            csv_col_name = dataset_map.get(dataset_folder_name)

            if not csv_col_name:
                print(f"  - Skipping folder '{dataset_folder_name}' as it's not in dataset_map.")
                continue

            lag_path = os.path.join(dataset_path, str(lag))

            for method_name, file_prefix in method_map.items():
                expected_filename_lower = f"{file_prefix}_time_lag{lag}.txt".lower()
                found_file_path = None
                result = 'TLE' # Default to TLE

                if os.path.isdir(lag_path):
                    for actual_filename in os.listdir(lag_path):
                        if actual_filename.lower() == expected_filename_lower:
                            found_file_path = os.path.join(lag_path, actual_filename)
                            break
                
                if found_file_path:
                    try:
                        with open(found_file_path, 'r') as f:
                            result = float(f.read().strip())
                    except (ValueError, IOError) as e:
                        print(f"  - Error reading {found_file_path}: {e}")
                        result = 'Error'
                
                # Store the result in the transposed structure
                lag_data[method_name][csv_col_name] = result

        # --- Create DataFrame and Save to CSV ---
        if any(lag_data.values()): # Check if any data was collected
            # Create DataFrame with methods as rows (index)
            df = pd.DataFrame.from_dict(lag_data, orient='index')
            
            df.index.name = 'Method'
            df = df.reset_index()

            # Order the dataset columns based on the user-defined list
            # Only include columns that were actually found and processed
            final_columns = ['Method'] + [ds for ds in dataset_order if ds in df.columns]
            df = df[final_columns]
            
            output_filename = f'running_times_lag{lag}.csv'
            df.to_csv(output_filename, index=False)
            print(f"✅ Successfully created '{output_filename}' with {len(df)} rows.")
        else:
            print(f"  - No data found for lag {lag}.")

if __name__ == '__main__':
    process_running_times()