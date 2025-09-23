"""
Core implementation of the TiMINo (Time series Models with Independent Noise)
causal discovery algorithm.

Based on the paper:
J. Peters, D. Janzing, B. Schoelkopf: "Causal Inference on Time Series using
Restricted Structural Equation Models" (NIPS 2013).
"""

import numpy as np
import warnings
from itertools import combinations
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.ar_model import AutoReg

# Suppress common warnings from statsmodels for cleaner output
warnings.filterwarnings("ignore")

## ---------------------------------------------------------------------------
## Part 1: Statistical Primitives
## ---------------------------------------------------------------------------

def traints_linear(x: np.ndarray, y: np.ndarray, pars: dict) -> dict:
    """
    Fits a linear time series model to predict x.
    Uses AutoReg for the univariate case and VAR for the multivariate case.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    max_lag = pars.get('maxOrder', 1)

    # --- Handle Univariate Case (x against its own past) ---
    if y.size == 0:
        if len(x) < max_lag + 1:
            raise ValueError(f"Time series length ({len(x)}) is too short for lag {max_lag}.")

        # Use AutoReg for single time series
        model = AutoReg(x, lags=max_lag)
        results = model.fit()

        # Pad residuals to match VAR output length for consistency
        residuals = np.full_like(x, np.nan, dtype=float)
        residuals[-len(results.resid):] = results.resid

        # AutoReg includes a trend/const by default, so order is number of lag params
        order = len(results.params) - 1 if model.trend == 'c' else len(results.params)

    # --- Handle Multivariate Case (x against its own past and y) ---
    else:
        x_reshaped = x.reshape(-1, 1)
        y_reshaped = y.reshape(x.shape[0], -1)
        data = np.hstack((x_reshaped, y_reshaped))

        if len(x) < max_lag + 1:
            raise ValueError(f"Time series length ({len(x)}) is too short for lag {max_lag}.")

        # Use VAR for multiple time series
        model = VAR(data)
        try:
            results = model.fit(maxlags=max_lag, ic=None)
            # The residuals array starts after the max_lag period
            residuals = np.full_like(x, np.nan, dtype=float)
            residuals[-len(results.resid):] = results.resid[:, 0]
            order = results.k_ar
        except np.linalg.LinAlgError:
            residuals = np.full_like(x, np.nan, dtype=float)
            order = max_lag

    # We need to return residuals aligned with the original x array's end
    # The first `order` elements will be NaN because they can't be predicted.
    # Let's align them properly. The models already exclude initial points.

    # The models in statsmodels return residuals for t > max_lag.
    # The original code did not pad, it just returned the shorter residual array.
    # Let's match that behavior for simplicity and directness.

    if y.size == 0:
         # AutoReg resid is for t = lags ... T-1
         final_residuals = results.resid
    else:
         # VAR resid is for t = k_ar ... T-1
         final_residuals = results.resid[:, 0]


    return {'resid': final_residuals, 'model': {'order': order}}


def indtestts_crosscov(x: np.ndarray, y: np.ndarray, alpha: float, max_lag: int, num_permutations: int = 200) -> dict:
    """
    Performs an independence test for time series based on cross-covariance
    using a permutation test.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if x.shape[0] != y.shape[0]:
        min_len = min(x.shape[0], y.shape[0])
        x, y = x[:min_len], y[:min_len]

    cross_corr = ccf(x, y, adjusted=True, fft=True)
    lags = np.arange(len(cross_corr))
    
    relevant_lags = np.concatenate((lags[1:max_lag + 1], lags[-(max_lag):]))
    if len(relevant_lags) == 0:
        return {'statistic': 0, 'crit.value': np.inf, 'p.value': 1.0}
        
    observed_ccf_values = cross_corr[relevant_lags]
    observed_stat = np.max(np.abs(observed_ccf_values)) if observed_ccf_values.size > 0 else 0

    perm_stats = []
    for _ in range(num_permutations):
        y_perm = np.random.permutation(y)
        perm_cross_corr = ccf(x, y_perm, adjusted=True, fft=True)
        perm_ccf_values = perm_cross_corr[relevant_lags]
        perm_stats.append(np.max(np.abs(perm_ccf_values)) if perm_ccf_values.size > 0 else 0)

    p_value = (np.sum(np.array(perm_stats) >= observed_stat) + 1) / (num_permutations + 1)
    crit_value = np.quantile(perm_stats, 1 - alpha)
    
    return {'statistic': observed_stat, 'crit.value': crit_value, 'p.value': p_value}


def traints_model(model, x, y, pars):
    return model(x, y, pars)

def indtestts(indtest, x, y, alpha, max_lag):
    return indtest(x, y, alpha, max_lag)

## ---------------------------------------------------------------------------
## Part 2: Algorithm Components
## ---------------------------------------------------------------------------

def fit_and_test_independence(x, y, z, alpha, max_lag, model, indtest, instant=1, output=False):
    """
    Fits models and tests for independence of residuals. This is the core
    computational step of the TiMINo algorithm.
    """
    min_lag = 4
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    # --- Fit model using only own past ---
    pars1 = {'maxOrder': max_lag}
    mod_fit1 = traints_model(model, x, [], pars1)
    order1 = max(mod_fit1['model']['order'], 1)
    r1 = mod_fit1['resid']
    
    z_aligned1 = z[order1:]
    
    t_quan = indtestts(indtest, z_aligned1, r1, alpha, max(min_lag, max_lag))
    
    if alpha > 0 and t_quan['p.value'] > alpha:
        if output:
            print("  - Univariate model yields independent residuals.")
        return {
            'statistic': t_quan['statistic'], 'crit.value': t_quan['crit.value'],
            'p.value': t_quan['p.value'], 'control': 1, 'order': mod_fit1['model']['order']
        }

    # --- If not independent, fit using y as a predictor ---
    x_s, y_s, z_s = x.copy(), y.copy(), z.copy()
    max_lag_s = max_lag
    if instant > 0:
        y_s = y_s[instant:]
        x_s = x_s[:-instant]
        z_s = z_s[:-instant]
        max_lag_s += instant

    pars2 = {'maxOrder': max_lag_s}
    mod_fit2 = traints_model(model, x_s, y_s, pars2)
    order2 = max(mod_fit2['model']['order'], 1)
    r2 = mod_fit2['resid']
    
    z_aligned2 = z_s[order2:]
    t_quan = indtestts(indtest, z_aligned2, r2, alpha, max(min_lag, max_lag + 1))

    return {
        'statistic': t_quan['statistic'], 'crit.value': t_quan['crit.value'],
        'p.value': t_quan['p.value'], 'control': 0, 'order': mod_fit2['model']['order']
    }

## ---------------------------------------------------------------------------
## Part 3: Main DAG Discovery Function
## ---------------------------------------------------------------------------

def timino_dag(M, alpha, max_lag, model=traints_linear, indtest=indtestts_crosscov,
               instant=0, confounder_check=0, output=False):
    """
    Discovers the causal graph (DAG) for a set of p time series using Algorithm 1
    [cite_start]from the TiMINo paper. [cite: 117]

    Args:
        M (np.ndarray): An (n_samples, p_features) array of time series data.
        alpha (float): The significance level for independence tests.
        max_lag (int): The maximum time lag to consider in models.
        model (callable): The function used for fitting time series models.
        indtest (callable): The function used for independence testing.
        instant (int): Lag for considering instantaneous effects.
        confounder_check (int): Not implemented. Placeholder for future extension.
        output (bool): If True, prints verbose progress updates.

    Returns:
        np.ndarray: A (p, p) adjacency matrix where C[i, j]=1 means i -> j.
                    Returns -1 for undecided relationships.
    """
    if confounder_check > 0:
        print("Warning: 'confounder_check' is not implemented in this version.")

    M = np.asarray(M)
    n_samples, p = M.shape
    C = np.zeros((p, p))
    
    S = list(range(p))
    causal_order = []
    
    while len(S) > 1:
        p_values, stats = {}, {}
        
        if output: print(f"\nSearching for sink in set: {S}")

        for k_idx, k_val in enumerate(S):
            parents = S[:k_idx] + S[k_idx+1:]
            
            if output: print(f"  - Testing {k_val} as sink with parents {parents}...")
            
            Fc = fit_and_test_independence(
                M[:, k_val], M[:, parents], M[:, parents],
                alpha, max_lag, model, indtest, instant, output
            )
            p_values[k_val], stats[k_val] = Fc['p.value'], Fc['statistic']

        possible_sinks = [node for node, p_val in p_values.items() if p_val > alpha]

        if not possible_sinks:
            if output: print(f"No possible sink node found in set {S}. Stopping search.")
            for i, j in combinations(S, 2):
                C[i, j] = C[j, i] = -1
            break
        
        best_sink = max(possible_sinks, key=lambda node: (p_values[node], -stats[node]))
        
        if output:
            print(f"--> Best sink found: {best_sink} (p-value={p_values[best_sink]:.3f})")

        causal_order.insert(0, best_sink)
        parents_of_sink = [node for node in S if node != best_sink]
        for parent in parents_of_sink:
            C[parent, best_sink] = 1
            
        S.remove(best_sink)

    if S: causal_order.insert(0, S[0])
    if output: print(f"\nFull causal order (source -> sink): {causal_order}")

    # --- Pruning unnecessary edges ---
    if alpha > 0 and np.any(C == 1):
        if output: print("\nPruning unnecessary edges...")
        for node in reversed(causal_order):
            parents = list(np.where(C[:, node] == 1)[0])
            if not parents: continue

            if output: print(f"  - Pruning parents of node {node}: {parents}")
            
            for parent_to_test in list(parents):
                remaining_parents = [p for p in parents if p != parent_to_test]
                target = M[:, node]
                
                if remaining_parents:
                    predictors = M[:, remaining_parents]
                    test_against = M[:, parents] 
                    Fc = fit_and_test_independence(
                        target, predictors, test_against, alpha,
                        max_lag, model, indtest, instant
                    )
                else:
                    mod_fit = traints_model(model, target, [], {'maxOrder': max_lag})
                    order = max(mod_fit['model']['order'], 1)
                    test_against = M[order:, [parent_to_test]]
                    Fc = indtestts(indtest, mod_fit['resid'], test_against.flatten(), alpha, max_lag)
                
                if Fc['p.value'] > alpha:
                    if output: print(f"    - Edge {parent_to_test} -> {node} is redundant. Pruning.")
                    C[parent_to_test, node] = 0
                    parents.remove(parent_to_test)
    
    return C