import pandas as pd
import numpy as np
import os

# Define file paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "Flood")
input_filename = DATA_PATH + "/rivers_ts_flood.csv" # Make sure this path is correct
output_filename = DATA_PATH + "/rivers_ts_flood_preprocessed.csv"


# --- Helper functions (Copied from previous context with robustness improvements) ---
def remove_trailing_nans(sample_prep):
    """
    Removes rows from the start and end of the DataFrame if they contain any NaNs
    after initial processing like interpolation.
    """
    if not isinstance(sample_prep, pd.DataFrame) or sample_prep.empty:
        return sample_prep
        
    first_valid_index = sample_prep.first_valid_index()
    last_valid_index = sample_prep.last_valid_index()

    if first_valid_index is None or last_valid_index is None:
        return pd.DataFrame(columns=sample_prep.columns)
    
    check_trailing_nans = np.array([])
    if not sample_prep.isnull().all().all():
        non_nan_rows_mask = ~sample_prep.isnull().any(axis=1)
        if non_nan_rows_mask.any():
             check_trailing_nans = np.where(non_nan_rows_mask)[0]

    if not len(check_trailing_nans) == 0:
        sample_prep = sample_prep.iloc[check_trailing_nans.min() : check_trailing_nans.max() + 1]
    elif not sample_prep.empty :
        pass

    if len(sample_prep) == 0 and not (first_valid_index is None and last_valid_index is None) :
        print("EMPTY SAMPLE DETECTED after remove_trailing_nans")
    return sample_prep


def preprocess_data(
    data,
    resolution="2H",
    interpolate=True,
    subset_year=False,
    subset_month=False, 
    subsample=1,
    normalize=False,
    remove_trailing_nans_early=False,
):
    sample_data = data.copy()

    if remove_trailing_nans_early:
        sample_data = remove_trailing_nans(sample_data)
        if sample_data.empty:
            print("Data became empty after early trailing NaN removal.")
            return sample_data

    if not isinstance(sample_data.index, pd.DatetimeIndex):
        try:
            sample_data.index = pd.to_datetime(sample_data.index)
        except Exception as e:
            print(f"Error converting index to datetime: {e}. Proceeding without resolution adjustment if index is not datetime.")
            pass

    if resolution and isinstance(sample_data.index, pd.DatetimeIndex):
        temp_dt_col = "_dt_temp_for_resample_"
        while temp_dt_col in sample_data.columns:
            temp_dt_col += "_"
        
        sample_data[temp_dt_col] = sample_data.index.round(resolution)
        numeric_cols = sample_data.select_dtypes(include=np.number).columns.tolist()
        
        cols_for_groupby = numeric_cols
        if not sample_data[numeric_cols].empty :
            sample_data = sample_data.groupby(temp_dt_col)[cols_for_groupby].mean()
        elif not sample_data.empty :
             sample_data = sample_data.groupby(temp_dt_col).first()

        if temp_dt_col in sample_data.columns and temp_dt_col not in numeric_cols :
             sample_data = sample_data.drop(columns=[temp_dt_col], errors='ignore')
        sample_data.index.name = data.index.name if data.index.name else "datetime"

    if subset_year: 
        if isinstance(sample_data.index, pd.DatetimeIndex):
            year_condition = sample_data.index.year == subset_year if isinstance(subset_year, int) else sample_data.index.year.isin(subset_year)
            if subset_month: 
                month_condition = sample_data.index.month.isin(subset_month)
                sample_data = sample_data.loc[year_condition & month_condition]
            else:
                sample_data = sample_data.loc[year_condition]
        else:
            print("Warning: Index is not DatetimeIndex, cannot subset by year/month.")

    if not sample_data.empty:
      sample_data = sample_data.iloc[::subsample, :]

    if normalize and not sample_data.empty:
        # Ensure only numeric columns are selected for min/max and normalization
        numeric_cols_to_normalize = sample_data.select_dtypes(include=np.number).columns
        if not numeric_cols_to_normalize.empty:
            data_min = sample_data[numeric_cols_to_normalize].min()
            data_max = sample_data[numeric_cols_to_normalize].max()
            diff = data_max - data_min
            for col in numeric_cols_to_normalize:
                if diff[col] == 0:
                    sample_data[col] = 0.0 
                else:
                    sample_data[col] = (sample_data[col] - data_min[col]) / diff[col]
    
    if interpolate and not sample_data.empty:
        # Interpolate only numeric columns
        numeric_cols_to_interpolate = sample_data.select_dtypes(include=np.number).columns
        if not numeric_cols_to_interpolate.empty:
            sample_data[numeric_cols_to_interpolate] = sample_data[numeric_cols_to_interpolate].interpolate(method='linear', axis=0, limit_direction='both')

    if not sample_data.empty:
        sample_data = remove_trailing_nans(sample_data)
    
    return sample_data


if __name__ == "__main__":

    # Check if input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file not found at {input_filename}")
        print("Please ensure the file exists in the specified 'causalriver' subdirectory or update the path.")
    else:
        print(f"Reading data from: {input_filename}")
        # Load data into pandas DataFrame
        try:
            df = pd.read_csv(input_filename, index_col='datetime')
            df.index = pd.to_datetime(df.index) # Ensure the index is a DatetimeIndex
            
            print("\nOriginal Data Head (first 5 rows):")
            print(df.head())
            print(f"\nOriginal Data Shape: {df.shape}")
            
            # Count missing values per column before preprocessing for non-empty columns
            missing_before = df.isnull().sum()
            missing_before = missing_before[missing_before > 0]
            if not missing_before.empty:
                print(f"\nMissing values per column before preprocessing (showing columns with NaNs):\n{missing_before}")
            else:
                print("\nNo missing values found in the original dataset.")


            # Define preprocessing configuration
            # These are example parameters; you might need to adjust them.
            preprocess_config = {
                'resolution': None,  # Resample to 1-hour intervals by taking the mean.
                # 'resolution': '2H',  # Resample to 1-hour intervals by taking the mean.
                # 'interpolate': True,   # Interpolate missing values using linear interpolation.
                'subset_year': False,  # No subsetting by year for this example.
                'subset_month': False, # No subsetting by month.
                'subsample': 1,        # No further subsampling of rows.
                'normalize': False,    # Set to True to normalize data (0-1 range).
                'remove_trailing_nans_early': True # Clean NaNs at start/end before major operations.
            }

            print(f"\n--- Preprocessing with config: {preprocess_config} ---")

            # Apply preprocessing
            df_preprocessed = preprocess_data(df, **preprocess_config)

            print("\nPreprocessed Data Head (first 5 rows):")
            print(df_preprocessed.head())
            print(f"\nPreprocessed Data Shape: {df_preprocessed.shape}")
            
            missing_after = df_preprocessed.isnull().sum()
            missing_after = missing_after[missing_after > 0]
            if not missing_after.empty:
                print(f"\nMissing values per column after preprocessing (showing columns with NaNs):\n{missing_after}")
            else:
                print("\nNo missing values found after preprocessing.")

            
            # Save the preprocessed data
            # Ensure the output directory exists if it's part of the path (not needed for "./")
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            df_preprocessed.to_csv(output_filename)
            print(f"\nPreprocessed data saved to: {output_filename}")

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_filename}")
        except Exception as e:
            print(f"An error occurred during processing: {e}")