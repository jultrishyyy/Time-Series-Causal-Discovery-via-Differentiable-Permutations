import os
import pandas as pd

def preprocess_csv_files(root_dir, paths):
    """
    Iterates through a list of CSV files, drops the first column,
    and overwrites the original file with the modified data.

    Args:
        root_dir (str): The absolute path to the project's root directory.
        paths (list): A list of lists, where each inner list contains
                      dataset folder, filename, and a prefix.
    """
    print("Starting to process CSV files...")

    for dataset_folder, filename in paths:
        # Construct the full path to the CSV file
        file_path = os.path.join(root_dir, "data", dataset_folder, filename.lstrip('/'))

        try:
            # Check if the file exists before processing
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: File not found, skipping: {file_path}")
                continue

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Check if the DataFrame has more than one column
            if df.shape[1] <= 1:
                print(f"ℹ️  Info: File has only one column, skipping: {file_path}")
                continue

            # Drop the first column (index 0)
            df_modified = df.iloc[:, 1:]

            # Save the modified DataFrame back to the original file path,
            # overwriting it without the index column.
            df_modified.to_csv(file_path, index=False)

            print(f"✅ Processed and overwritten: {file_path}")

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    # List of datasets to process
    paths_to_process = [
        ['Middleware_oriented_message_Activity', '/monitoring_metrics_1.csv'],
        ['Middleware_oriented_message_Activity', '/monitoring_metrics_2.csv'],
        # ['Storm_Ingestion_Activity', '/storm_data_normal.csv'],
        # ['Web_Activity', '/preprocessed_1.csv'],
        # ['Web_Activity', '/preprocessed_2.csv'],
        # ['Antivirus_Activity', '/preprocessed_1.csv'],
        # ['Antivirus_Activity', '/preprocessed_2.csv'],
        # ['Flood', '/rivers_ts_flood_preprocessed.csv'],
        # ['Bavaria', '/rivers_ts_bavaria_preprocessed.csv'],
        # ['East_germany', '/rivers_ts_east_germany_preprocessed.csv'],
        # ['SWaT', '/preprocessed_1.csv'],
    ]

    # Define the root directory. This assumes the script is run from a directory
    # that is a sibling to the 'data' directory (e.g., from a 'scripts' folder).
    # Adjust if your project structure is different.
    try:
        # Assumes the script is in a subfolder of the project root
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    except NameError:
        # Fallback for interactive environments like Jupyter
        ROOT_DIR = os.path.abspath('.')


    # Run the preprocessing function
    preprocess_csv_files(ROOT_DIR, paths_to_process)
