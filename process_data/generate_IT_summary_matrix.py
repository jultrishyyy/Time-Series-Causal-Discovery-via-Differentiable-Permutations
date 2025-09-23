import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


# Modify the paths to your input and output files
DATA_PATH = os.path.join(ROOT_DIR, "data", "Storm_Ingestion_Activity")
input_data_filename = DATA_PATH + '/storm_data_normal.csv'
input_structure_filename = DATA_PATH + '/structure.txt'
output_matrix_filename = DATA_PATH + '/summary_matrix.npy'
output_graph_filename = DATA_PATH + '/causal_graph.png'


if __name__ == "__main__":

    print(f"\nInput data file: {input_data_filename}")
    print(f"\nInput structure file: {input_structure_filename}")
    param_data = pd.read_csv(input_data_filename, delimiter=',', index_col=0, header=0)

    param_data.columns = param_data.columns.str.replace(' ', '_')
    print("\nParameter data:")
    print(param_data.columns)

    three_col_format = np.loadtxt(input_structure_filename, delimiter=' ', dtype=str)
    print(three_col_format)

    summary_matrix = pd.DataFrame(np.zeros([param_data.shape[1], param_data.shape[1]]), columns=param_data.columns, index=param_data.columns, dtype=int)

    for i in range(three_col_format.shape[0]):
        c = three_col_format[i, 0]
        e = three_col_format[i, 1]
        summary_matrix.loc[e, c] = 1

    print("\nSummary adjacency matrix:")
    print(summary_matrix)

    summary_matrix_np = summary_matrix.values
    # np.fill_diagonal(summary_matrix_np, 1)
    try:
        np.save(output_matrix_filename, summary_matrix_np)
        print(f"\nSummary matrix (as NumPy array) saved to '{output_matrix_filename}'")
    except Exception as e:
        print(f"Error saving summary matrix as .npy: {e}")
