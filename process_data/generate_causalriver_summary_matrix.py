import sys
import os
import pickle
import numpy as np
import pandas as pd
import graphviz
import networkx as nx # For topological sort# 1. Load Preprocessed Data



# Define file paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "Flood")
input_data_filename = DATA_PATH + "/rivers_ts_flood_preprocessed.csv" # Make sure this path is correct
input_label_filename = DATA_PATH + "/rivers_flood.p" # Ground truth graph data
output_matrix_filename = DATA_PATH + '/summary_matrix.npy'



import pandas as pd
import numpy as np
import pickle
import sys
import os

def create_summary_matrix_from_graph_file(data_csv_filepath, graph_pickle_filepath):
    """
    Generates a summary adjacency matrix from a ground truth graph file.
    The convention for the summary matrix is: Matrix.loc[effect_node, cause_node] = 1.
    This means rows represent the EFFECT, and columns represent the CAUSE.

    Args:
        data_csv_filepath (str): Path to the CSV file containing the data.
                                 The column order (after handling any index column like 'datetime')
                                 defines the matrix node order for rows (effects) and columns (causes).
        graph_pickle_filepath (str): Path to the .p file containing the graph object
                                     (e.g., NetworkX graph or similar with .edges attribute).
                                     The graph nodes are expected to be strings or convertible to strings
                                     that match the column names from data_csv_filepath.

    Returns:
        tuple: (pandas.DataFrame, list)
               - The summary adjacency matrix (node names as index/effects, columns/causes).
               - A list of edges from the loaded graph (e.g., [('cause1', 'effect1'), ...]).
               Returns (None, None) if a significant error occurs.
    """
    # --- 1. Load Data Columns to determine node order ---
    node_names_ordered = []
    try:
        # Check if 'datetime' column exists to use as index_col
        header_df = pd.read_csv(data_csv_filepath, nrows=0)
        index_col_name = 'datetime' if 'datetime' in header_df.columns else None
        
        temp_df_for_columns = pd.read_csv(data_csv_filepath, index_col=index_col_name)
        node_names_ordered = list(temp_df_for_columns.columns)
        
        print(f"Node order successfully extracted from data file '{data_csv_filepath}'.")
        print(f"Nodes (will be used for row and column names): {node_names_ordered}")
        if not node_names_ordered:
            print(f"Error: No data columns (nodes) found in {data_csv_filepath} after handling index.")
            return None, None
    except FileNotFoundError:
        print(f"Error: Data CSV file not found at '{data_csv_filepath}'")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: Data CSV file at '{data_csv_filepath}' is empty.")
        return None, None
    except Exception as e:
        print(f"Error loading or processing data columns from '{data_csv_filepath}': {e}")
        return None, None

    num_nodes = len(node_names_ordered)
    node_to_idx = {node_name: i for i, node_name in enumerate(node_names_ordered)}

    # --- 2. Load Ground Truth Graph from .p file ---
    graph_sample = None
    loaded_graph_edges = [] # To store edges for verification
    try:
        with open(graph_pickle_filepath, 'rb') as f:
            ground_truth_data_loaded = pickle.load(f)
        
        if isinstance(ground_truth_data_loaded, list):
            if len(ground_truth_data_loaded) > 0:
                graph_sample = ground_truth_data_loaded[0]
                print(f"Successfully loaded ground truth: list of graphs found in '{graph_pickle_filepath}', using the first graph.")
            else:
                print(f"Error: Graph file '{graph_pickle_filepath}' contained an empty list.")
                return None, None
        elif hasattr(ground_truth_data_loaded, 'edges') and hasattr(ground_truth_data_loaded, 'nodes'):
            graph_sample = ground_truth_data_loaded
            print(f"Successfully loaded ground truth: single graph object found in '{graph_pickle_filepath}'.")
        else:
            print(f"Error: Graph file '{graph_pickle_filepath}' does not contain a recognizable graph format. "
                  f"Expected a list of graphs or a single graph object with 'edges' and 'nodes' attributes. "
                  f"Type found: {type(ground_truth_data_loaded)}")
            return None, None

        if graph_sample is None:
            print(f"Critical Error: Graph sample could not be determined from '{graph_pickle_filepath}'.")
            return None, None
            
        print(f"Ground truth graph object type: {type(graph_sample)}")
        
        if hasattr(graph_sample, 'edges'):
            try:
                loaded_graph_edges = list(graph_sample.edges)
            except Exception as e:
                print(f"Warning: Could not directly convert graph_sample.edges to a list: {e}. Edges might not be available for verification.")
                loaded_graph_edges = [] # Ensure it's a list
        else:
            print("Warning: Loaded graph object does not have an 'edges' attribute. Cannot get edges for verification.")


    except FileNotFoundError:
        print(f"Error: Graph pickle file not found at '{graph_pickle_filepath}'")
        return None, None
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle graph file at '{graph_pickle_filepath}'. File might be corrupted or not a pickle file.")
        return None, None
    except Exception as e:
        print(f"Error loading graph from '{graph_pickle_filepath}': {e}")
        return None, None

    # --- 3. Initialize Adjacency Matrix ---
    summary_adj_matrix_df = pd.DataFrame(np.zeros((num_nodes, num_nodes), dtype=int),
                                         index=node_names_ordered,   # Rows (effects)
                                         columns=node_names_ordered) # Columns (causes)

    # --- 4. Populate Adjacency Matrix ---
    if not loaded_graph_edges and hasattr(graph_sample, 'edges'): # Attempt to get edges again if initial try failed but attribute exists
        try:
            loaded_graph_edges = list(graph_sample.edges)
        except:
            pass # Keep it as an empty list if it still fails

    edges_processed_count = 0
    unmapped_edges = []
    
    if not loaded_graph_edges and not hasattr(graph_sample, 'edges'):
        print(f"Error: The loaded graph object (type: {type(graph_sample)}) does not have an 'edges' attribute to iterate over.")
        return summary_adj_matrix_df, loaded_graph_edges # Return matrix (all zeros) and empty edges

    for u, v in loaded_graph_edges: # Edge u -> v (u is CAUSE, v is EFFECT)
        u_str = str(u) 
        v_str = str(v) 

        if u_str in node_to_idx and v_str in node_to_idx:
            idx_u_cause = node_to_idx[u_str]   
            idx_v_effect = node_to_idx[v_str]  
            
            summary_adj_matrix_df.iloc[idx_v_effect, idx_u_cause] = 1
            edges_processed_count += 1
        else:
            unmapped_edges.append(f"({u_str} -> {v_str})")
            if u_str not in node_to_idx:
                 print(f"  Warning: Cause node '{u_str}' from edge ({u_str} -> {v_str}) not found in data columns.")
            if v_str not in node_to_idx:
                 print(f"  Warning: Effect node '{v_str}' from edge ({u_str} -> {v_str}) not found in data columns.")

    print(f"\nProcessed {edges_processed_count} edges from the graph and mapped them to the matrix.")
    if unmapped_edges:
        print(f"Warning: {len(unmapped_edges)} edge(s) could not be fully mapped as one or both nodes were not found in the data columns:")
        for i, edge_str in enumerate(unmapped_edges):
            if i < 5: 
                print(f"  - {edge_str}")
        if len(unmapped_edges) > 5:
            print(f"  ... and {len(unmapped_edges) - 5} more unmapped edges.")
        print(f"  Available data columns (nodes): {node_names_ordered}")
        print(f"  Example nodes from graph (first 5, if available):")
        try:
            graph_nodes_list = list(graph_sample.nodes)
            print(f"    {graph_nodes_list[:5]}")
        except Exception:
            print("    Could not list nodes from graph object.")

    if edges_processed_count == 0 and len(loaded_graph_edges) > 0:
        print(f"Major Warning: No edges from the graph (total {len(loaded_graph_edges)}) were mapped to the matrix. "
              "This usually indicates a mismatch between node names in the graph file and column names in the data CSV file.")
    
    return summary_adj_matrix_df, loaded_graph_edges



if __name__ == "__main__":
    
    print(f"\n--- Generating Summary Matrix (Effect=Row, Cause=Column) ---")
    print(f"Using data file: {input_data_filename}")
    print(f"Using graph (label) file: {input_label_filename}")
    
    summary_matrix, graph_edges_for_verification = create_summary_matrix_from_graph_file(
        input_data_filename, input_label_filename
    )

    if summary_matrix is not None:
        print("\n--- Generated Summary Adjacency Matrix ---")
        print("(Convention: Matrix.loc[effect_node, cause_node] = 1)")
        with pd.option_context('display.max_rows', 15, 'display.max_columns', 15, 'display.width', 150):
             print(summary_matrix)

        # --- Verification of matrix against some graph edges ---
        print("\n--- Verifying Matrix Against Graph Edges ---")
        if graph_edges_for_verification and not summary_matrix.empty:
            num_edges_to_check = min(len(graph_edges_for_verification), 5) # Check up to 5 edges
            if num_edges_to_check == 0:
                print("No edges found in the graph to verify against.")
            else:
                print(f"Checking the first {num_edges_to_check} edge(s) from the loaded graph:")
                for i in range(num_edges_to_check):
                    u, v = graph_edges_for_verification[i]
                    u_str, v_str = str(u), str(v) # u is cause, v is effect
                    
                    print(f"  Edge {i+1}: {u_str} -> {v_str} (Cause: {u_str}, Effect: {v_str})")
                    
                    # Check if nodes are in matrix dimensions before accessing .loc
                    if v_str in summary_matrix.index and u_str in summary_matrix.columns:
                        val = summary_matrix.loc[v_str, u_str]
                        print(f"    summary_matrix.loc['{v_str}', '{u_str}'] = {val}. Expected: 1.")
                        if val != 1:
                            print(f"      Mismatch for edge {u_str} -> {v_str}!")
                    else:
                        print(f"    Node(s) for edge {u_str} -> {v_str} not found in matrix dimensions. "
                              f"(Effect '{v_str}' in rows? {v_str in summary_matrix.index}. "
                              f"Cause '{u_str}' in columns? {u_str in summary_matrix.columns})")
        elif summary_matrix.empty and graph_edges_for_verification:
             print("Matrix is empty, cannot verify edges.")
        else:
            print("No graph edges were loaded, cannot perform verification.")

        try:
            summary_matrix_np = summary_matrix.values 
            np.save(output_matrix_filename, summary_matrix_np)
            print(f"Summary matrix (as NumPy array) successfully saved to: '{output_matrix_filename}'")
        except Exception as e:
            print(f"Error saving summary matrix as .npy: {e}")
    else:
        print("\n--- Failed to generate summary matrix ---")
        print("Please check the error messages above for details.")

