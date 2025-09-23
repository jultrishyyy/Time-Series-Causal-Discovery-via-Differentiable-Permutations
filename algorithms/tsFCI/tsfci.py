"""
Core implementation of the tsFCI algorithm for causal discovery from time series data.

This version uses the SciPy library for conditional independence testing to avoid
pgmpy versioning issues.

Based on the paper:
Entner, D., & Hoyer, P. O. (2010). On Causal Discovery from Time Series Data using FCI.
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations
from scipy.stats import chi2_contingency
import re

class IndependenceTester:
    """
    A class to perform conditional independence tests using SciPy's chi2_contingency.
    This test is suitable for discrete data.
    """
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.data = data
        self.alpha = alpha

    def is_independent(self, X: str, Y: str, Z: list[str] = None) -> bool:
        """
        Checks if variable X is conditionally independent of Y given a set of variables Z.
        Returns True if the p-value is greater than the significance level.
        """
        if Z is None:
            Z = []
        
        if not Z:
            # --- Unconditional Independence Test ---
            contingency_table = pd.crosstab(self.data[X], self.data[Y])
            # The g-test (likelihood ratio test) is equivalent to 'gsq'
            g_stat, p_value, dof, expected = chi2_contingency(contingency_table, lambda_="log-likelihood")

        else:
            # --- Conditional Independence Test ---
            # We calculate a weighted sum of G-tests for each configuration of Z
            total_g_stat = 0
            total_dof = 0
            
            # If Z is a list with one item, pass it as a string to avoid the warning
            grouper = Z[0] if len(Z) == 1 else Z

            # Group data by the state of conditioning variables
            for z_state, group in self.data.groupby(grouper):
                if len(group) < 2:
                    continue
                contingency_table = pd.crosstab(group[X], group[Y])
                
                # Skip tables that are too small for a meaningful test
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                    continue

                g_stat, _, dof, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
                total_g_stat += g_stat
                total_dof += dof
                
            if total_dof == 0:
                # Cannot perform the test if no valid contingency tables were found
                return True 
                
            # The sum of G-statistics is chi-squared distributed with the sum of dof
            from scipy.stats import chi2
            p_value = 1.0 - chi2.cdf(total_g_stat, total_dof)

        return p_value > self.alpha

def get_time_slice(var_name: str) -> int:
    """
    Extracts the time slice 't' from a variable name like 'V1_t-1'.
    Uses regex for robustness against column names containing '_t'.
    """
    # Regex to find '_t' followed by an optional '-' and digits at the end of the string
    match = re.search(r'_t(-?\d*)$', var_name)
    if not match:
        return 0  # Not a time-lagged variable name

    time_str = match.group(1)
    if not time_str:  # Matches '_t' at the end
        return 0

    try:
        # The part after '_t' is the lag
        return int(time_str)
    except ValueError:
        # Should not happen with this regex, but as a fallback
        return 0

def get_homologous_edges(u: str, v: str, all_vars: list[str], max_lag: int) -> list[tuple[str, str]]:
    """
    Finds all edges homologous to the edge (u, v) based on time-invariance.
    This version uses rsplit to handle variable names containing '_t'.
    """
    # Use rsplit('_t', 1) to split only on the last occurrence of '_t'
    base_u, time_u_str = u.rsplit('_t', 1)
    base_v, time_v_str = v.rsplit('_t', 1)
    
    time_u = 0 if not time_u_str else int(time_u_str)
    time_v = 0 if not time_v_str else int(time_v_str)
    lag_diff = time_u - time_v
    
    homologous = []
    for t_offset in range(-max_lag, 1):
        new_u = f"{base_u}_t{t_offset}" if t_offset != 0 else f"{base_u}_t"
        new_v = f"{base_v}_t{t_offset - lag_diff}" if (t_offset - lag_diff) != 0 else f"{base_v}_t"

        if new_u in all_vars and new_v in all_vars:
            homologous.append(tuple(sorted((new_u, new_v))))
            
    return list(set(homologous))

def create_time_lagged_dataset(data: np.ndarray, var_names: list[str], max_lag: int) -> pd.DataFrame:
    """
    Transforms a time series dataset into a time-lagged, i.i.d.-like dataset
    in a memory-efficient way to avoid fragmentation.
    """
    n_samples, n_vars = data.shape
    lagged_data_dict = {}

    for lag in range(max_lag + 1):
        shifted_data = np.roll(data, lag, axis=0)
        if lag > 0:
            shifted_data[:lag, :] = np.nan

        for i, var_name in enumerate(var_names):
            col_name = f"{var_name}_t" if lag == 0 else f"{var_name}_t-{lag}"
            lagged_data_dict[col_name] = shifted_data[:, i]

    # Create the DataFrame from the dictionary all at once
    df_lagged = pd.DataFrame(lagged_data_dict)

    # Drop rows with NaN values introduced by shifting
    df_lagged.dropna(inplace=True)
    df_lagged.reset_index(drop=True, inplace=True)
    
    return df_lagged.astype(int)

def tsfci(data: np.ndarray, var_names: list[str], max_lag: int, alpha: float = 0.05, verbose: bool = False) -> dict:
    """Implementation of the tsFCI algorithm."""
    if verbose: print("Step 1: Creating time-lagged dataset...")
    lagged_data = create_time_lagged_dataset(data, var_names, max_lag)
    all_vars = sorted(lagged_data.columns.tolist(), key=get_time_slice, reverse=True)
    
    tester = IndependenceTester(lagged_data, alpha)

    if verbose: print("\nStep 2: Finding the skeleton...")
    adj = {v: [o for o in all_vars if o != v] for v in all_vars}
    sepsets = {}

    for l in range(len(all_vars)):
        for u, v in permutations(all_vars, 2):
            if v not in adj.get(u, []):
                continue
            
            time_u, time_v = get_time_slice(u), get_time_slice(v)
            potential_Z = [n for n in adj[u] if n != v and get_time_slice(n) < min(time_u, time_v)]
            
            if len(potential_Z) >= l:
                for Z in combinations(potential_Z, l):
                    Z = list(Z)
                    if tester.is_independent(u, v, Z):
                        homologous_edges = get_homologous_edges(u, v, all_vars, max_lag)
                        for hu, hv in homologous_edges:
                            if hv in adj.get(hu, []):
                                adj[hu].remove(hv)
                                adj[hv].remove(hu)
                                sepsets[(hu, hv)] = Z
                                sepsets[(hv, hu)] = Z
                        if verbose: print(f"  - Removed edge {u}-{v} (and homologous) based on CI: {u} indep. {v} | {Z}")
                        break
            if v not in adj.get(u, []):
                continue

    if verbose: print("\nStep 3: Orienting edges...")
    pag = {v: {'->': [], '<-': [], '<->': []} for v in all_vars}
    for u, neighbors in adj.items():
        time_u = get_time_slice(u)
        for v in neighbors:
            time_v = get_time_slice(v)
            if time_u > time_v and u not in pag[v]['<-']:
                pag[u]['<-'].append(v)
                pag[v]['->'].append(u)
            elif time_u == time_v and u not in pag[v]['<->'] and v not in pag[u]['<->']:
                pag[u]['<->'].append(v)
    
    return adj, sepsets