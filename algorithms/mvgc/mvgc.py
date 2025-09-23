import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

def MVGC(data: pd.DataFrame, max_lags: int, sig_level: float = 0.05, scale: bool = True):
    """
    Performs a Multivariate Granger Causality test on a set of time series.

    Args:
        data (pd.DataFrame): A DataFrame where each column is a time series and the index is time.
        max_lags (int): The maximum number of lags to consider for the VAR model.
        sig_level (float): The significance level (alpha) for the F-test. Causal links
                           are only accepted if the p-value is below this threshold.
        scale (bool): If True, standardizes the data before analysis.
        verbose (bool): If True, prints detailed test results for each pair.

    Returns:
        pd.DataFrame: An adjacency matrix where a value of 1 at [row, col] indicates
                      that the column variable Granger-causes the row variable.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    variables = data.columns.tolist()
    df = data.copy()

    # --- 1. Data Preprocessing ---
    if scale:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df.values)
        df = pd.DataFrame(df_scaled, columns=variables, index=data.index)

    # --- 2. Fit the Vector Autoregressive (VAR) Model ---
    # The model finds the optimal lag order up to max_lags based on AIC.
    model = VAR(df)
    try:
        results = model.fit(maxlags=max_lags, ic='aic')
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        # Return an empty matrix if the model fails to fit
        return pd.DataFrame(np.zeros((len(variables), len(variables))), index=variables, columns=variables)

    # --- 3. Perform Granger Causality Tests ---
    # Create an empty DataFrame to store the results (p-values)
    p_value_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), index=variables, columns=variables)
    
    # Iterate over all possible pairs of variables
    for caused_var in variables:       # The potential effect (Y)
        for causing_var in variables:  # The potential cause (X)
            if caused_var == causing_var:
                continue  # Skip self-causality test

            # Test the null hypothesis that 'causing_var' does NOT Granger-cause 'caused_var'
            test_result = results.test_causality(
                caused=caused_var,
                causing=causing_var,
                kind='f', # Use the F-test
                signif=sig_level
            )
            
            p_value = test_result.pvalue
            p_value_matrix.loc[caused_var, causing_var] = p_value


    # --- 4. Create the Final Adjacency Matrix ---
    # A value of 1 means the null hypothesis was rejected (i.e., there is a causal link)
    adjacency_matrix = (p_value_matrix < sig_level).astype(int)
    
    return adjacency_matrix