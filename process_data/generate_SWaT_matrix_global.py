import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


# Modify the paths to your input and output files
DATA_PATH = os.path.join(ROOT_DIR, "data", "SWaT")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "SWaT")
input_data_filename = OUTPUT_PATH + '/preprocessed_2.csv'
input_structure_filename = DATA_PATH + '/ground_truth.txt'


if __name__ == "__main__":
    with open(input_structure_filename, 'r') as f:
        file_content = f.read()
    X = pd.read_csv(input_data_filename, delimiter=',', header=0)
    column_names = X.columns.tolist()
    print("Length of column names:", len(column_names))

    causal_pairs = set()
    for line in file_content.strip().split('\n'):
        if line:
            cause, effect = line.strip().split(',')
            causal_pairs.add((cause, effect))
    # print(f"Total causal pairs found:")
    # print(causal_pairs)


    # Create an empty N x N matrix with zeros, labeled with the column names
    summary_matrix = pd.DataFrame(0, index=column_names, columns=column_names)

    # Populate the matrix: if a cause->effect link exists, set the cell to 1
    for cause, effect in causal_pairs:
        # Check if both cause and effect are in our list of columns to avoid errors
        if cause in summary_matrix.columns and effect in summary_matrix.index:
            summary_matrix.loc[effect, cause] = 1


    print(summary_matrix)

    # Convert the DataFrame to a NumPy array
    numpy_matrix = summary_matrix.to_numpy()

    print("Numpy matrix shape:", numpy_matrix.shape)
    
    # Define a unique output filename for this stage
    filename = OUTPUT_PATH + f"/summary_matrix.npy"
    
    # Save the NumPy array to its own .npy file
    np.save(filename, numpy_matrix)
    
    print(f"✅ Successfully generated and saved '{filename}'")



