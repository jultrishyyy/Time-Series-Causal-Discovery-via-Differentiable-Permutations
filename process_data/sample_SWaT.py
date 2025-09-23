import pandas as pd
import io
import numpy as np
import os
import sys 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "SWaT")


# data_filename = DATA_PATH + '/SWaT_Dataset_Normal_Part0.csv'
data_filename = DATA_PATH + '/consitent_values_removed.csv'
output_filename = DATA_PATH + '/consitent_values_removed_sampled.csv'


def sample_data_with_pandas(data_file, frequency, output_filename):
    """
    Samples data from a multi-line string using pandas and saves it to a CSV file.

    Args:
        data_string (str): The input data as a single string with newline separators.
        frequency (int): The sampling frequency (e.g., 5 for every 5th row).
        output_filename (str): The name of the CSV file to save the output.
    """

    
    # Read the data into a pandas DataFrame
    try:
        df = pd.read_csv(data_file, header=0)
    except Exception as e:
        print(f"Error reading data into DataFrame: {e}")
        return

    # Sample the DataFrame using iloc for integer-location based indexing.
    # The logic is the same: start at index `frequency - 1` and step by `frequency`.
    sampled_df = df.iloc[frequency-1::frequency]
    
    # Write the resulting DataFrame to a new CSV file.
    # `index=False` prevents pandas from writing the DataFrame index as a column.
    try:
        sampled_df.to_csv(output_filename, index=False)
        print(f"Successfully saved sampled data to '{output_filename}'")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # Return the sampled data as a string for printing
    return sampled_df.to_csv(index=False)


# Set the desired sampling frequency
sampling_frequency = 5

# Call the function to sample the data and save it to a file
sampled_output = sample_data_with_pandas(data_filename, sampling_frequency, output_filename)

# print("Sampled Data:")
# print(sampled_output.head())

