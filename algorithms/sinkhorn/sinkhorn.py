import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Optional
import random
from copy import deepcopy
from scipy.optimize import linear_sum_assignment

# --- The Main Model ---
class PermutationSVAR(nn.Module):
    """
    The core PyTorch model for discovering causal structure using a
    Structural Vector Autoregressive (SVAR) model with a permutation-based
    approach to ensure acyclicity.
    """
    def __init__(self, n_features: int, max_lags: int):
        super().__init__()
        self.n_features = n_features
        self.max_lags = max_lags
        self.permutation_logits = nn.Parameter(torch.randn(n_features, n_features))
        self.inst_regressor_weight = nn.Parameter(torch.randn(n_features, n_features))
        self.lagged_regressor = nn.Linear(n_features * max_lags, n_features, bias=False)

# --- The Main User-Facing Class ---
class Sinkhorn:
    """
    A unified class to handle unsupervised causal discovery using the PermutationSVAR model.
    It encapsulates data preprocessing and training.
    """

    def __init__(self,
                 max_lags: int = 5,
                 l1_inst_strength: float = 0.001,
                 l1_lagged_strength: float = 0.001,
                 early_stopping_patience: int = 200,
                 random_seed: Optional[int] = 42,
                 save_path: Optional[str] = None):
        """
        Initializes the configuration for the causal discovery model.

        Args:
            max_lags (int): Maximum number of time lags.
            l1_inst_strength (float): L1 regularization weight for instantaneous effects.
            l1_lagged_strength (float): L1 regularization weight for lagged effects.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
            random_seed (Optional[int]): Seed for reproducibility. If None, the run will be stochastic.
            save_path (Optional[str]): Path to save the trained model after fitting.
        """
        self.max_lags = max_lags
        self.l1_inst_strength = l1_inst_strength
        self.l1_lagged_strength = l1_lagged_strength
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        self.save_path = save_path

        self.model: Optional[PermutationSVAR] = None
        self.adjacency_matrices: List[np.ndarray] = []
        self.n_features: Optional[int] = None
        self.causal_order: Optional[List[int]] = None
        self.scaler: Optional[StandardScaler] = None

    @staticmethod
    def _set_random_seeds(seed_value: int):
        """Sets random seeds for reproducibility across all relevant libraries."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _prepare_data(self, X_numpy: np.ndarray):
        """Scales the data and creates lagged tensors for training."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numpy)
        self.scaler = scaler
        
        n_samples = X_scaled.shape[0]
        
        Y_data = X_scaled[self.max_lags:]
        X_lagged = np.zeros((n_samples - self.max_lags, self.n_features * self.max_lags))

        for t in range(self.max_lags, n_samples):
            X_lagged[t - self.max_lags, :] = X_scaled[t - self.max_lags : t, :][::-1].flatten()

        self.Y_tensor = torch.from_numpy(Y_data).float()
        self.X_lagged_tensor = torch.from_numpy(X_lagged).float()
        print(f"Target tensor shape: {self.Y_tensor.shape}")
        print(f"Lagged features tensor shape: {self.X_lagged_tensor.shape}")

    @staticmethod
    def _sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm to produce a doubly stochastic matrix."""
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        return log_alpha.exp()

    @staticmethod
    def _gumbel_sinkhorn(log_alpha: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Gumbel-Sinkhorn for producing soft or hard permutation matrices."""
        if hard:
            # For hard permutation, we use the Hungarian algorithm (linear_sum_assignment)
            # on the learned logits for optimal assignment.
            logits_np = log_alpha.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-logits_np) # Use negative for maximization
            hard_perm_matrix = torch.zeros_like(log_alpha)
            hard_perm_matrix[row_ind, col_ind] = 1
            return hard_perm_matrix
        
        # For soft permutation, apply Gumbel noise and the Sinkhorn algorithm.
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_alpha) + 1e-20) + 1e-20)
        log_alpha_gumbel = (log_alpha + gumbel_noise) / temperature
        return Sinkhorn._sinkhorn(log_alpha_gumbel)

    def _calculate_loss(self, X_lagged: torch.Tensor, Y_target: torch.Tensor, P_soft: torch.Tensor):
        """
        Calculates the total loss, which includes reconstruction and L1 regularization.
        """
        I_tensor = torch.eye(self.n_features, device=Y_target.device)
        mse_loss_fn = nn.MSELoss()
        
        # Enforce acyclicity in the instantaneous effects matrix (B0)
        B0_dense = self.model.inst_regressor_weight
        B0_permuted = P_soft @ B0_dense @ P_soft.T
        B0_permuted_lower = torch.tril(B0_permuted, diagonal=-1)
        B0_acyclic = P_soft.T @ B0_permuted_lower @ P_soft
        
        # Calculate the reconstruction error based on the SVAR model equation
        LHS = Y_target @ (I_tensor - B0_acyclic).T
        RHS = self.model.lagged_regressor(X_lagged)
        reconstruction_loss = mse_loss_fn(LHS, RHS)

        # Calculate the L1 sparsity-inducing penalty
        l1_loss = self.l1_inst_strength * torch.sum(torch.abs(B0_dense)) + \
                  self.l1_lagged_strength * torch.sum(torch.abs(self.model.lagged_regressor.weight))
        
        total_loss = reconstruction_loss + l1_loss
        
        return total_loss, reconstruction_loss, l1_loss

    def fit(self, X: Union[pd.DataFrame, np.ndarray], epochs: int = 10000, lr: float = 0.005):
        """
        Fits the model to the provided data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input time-series data.
            epochs (int): The number of training epochs.
            lr (float): The learning rate.
        """
        if self.random_seed is not None:
            self._set_random_seeds(self.random_seed)
            print(f"Random seed set to {self.random_seed} for reproducibility.")
        
        print(f"--- Starting unsupervised training for {epochs} epochs with lr={lr} ---")
        
        X_numpy = np.asarray(X)
        _, self.n_features = X_numpy.shape
        print(f"Data received: {X_numpy.shape[0]} samples, {self.n_features} features.")
        
        self.model = PermutationSVAR(self.n_features, self.max_lags)
        print(f"Model initialized with {self.n_features} features and {self.max_lags} maximum lags.")
        
        self._prepare_data(X_numpy)
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Anneal temperature for Gumbel-Sinkhorn from a high to a low value
        temperature_schedule = np.linspace(5.0, 0.1, epochs)
        best_total_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            temp = temperature_schedule[epoch]
            P_soft = self._gumbel_sinkhorn(self.model.permutation_logits, temperature=temp)

            # Calculate training loss
            total_loss, _, _ = self._calculate_loss(self.X_lagged_tensor, self.Y_tensor, P_soft)

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch+1:5d}/{epochs} | Train Loss: {total_loss.item():.4f}")

            # --- Early Stopping Check based on TRAINING LOSS ---
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                patience_counter = 0
                best_model_state = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"\n--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Training loss did not improve for {self.early_stopping_patience} epochs.")
                break

        print("\n--- Training finished ---")
        
        # Load the best model state found during training
        if best_model_state:
            print(f"Restoring model to best performance (Loss: {best_total_loss:.4f}).")
            self.model.load_state_dict(best_model_state)
        
        self._extract_adjacency_matrices()

        if self.save_path:
            # Save the state of the *best* model
            if best_model_state:
                torch.save(best_model_state, self.save_path)
                print(f"Best model state saved to {self.save_path}")
            else: # Fallback if training was too short
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Final model state saved to {self.save_path}")

    def _extract_adjacency_matrices(self):
        """Extracts, prunes, and stores the final adjacency matrices."""
        self.model.eval()
        
        # Use the Hungarian algorithm on the final logits to get the optimal permutation
        logits = self.model.permutation_logits.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-logits)
        
        # The col_ind gives the causal order. `causal_order[i] = j` means variable j is in the i-th position.
        causal_order = np.zeros(self.n_features, dtype=int)
        causal_order[row_ind] = col_ind
        self.causal_order = causal_order.tolist()
        print(f"Discovered causal order: {self.causal_order}")
        
        # Create a causal mask to enforce the discovered acyclic structure
        causal_mask = np.zeros((self.n_features, self.n_features))
        for i in range(1, self.n_features):
            causal_mask[causal_order[i], causal_order[:i]] = 1.0
            
        B0_est = self.model.inst_regressor_weight.detach().numpy() * causal_mask

        # Extract the lagged coefficient matrices
        B_lagged_stack = self.model.lagged_regressor.weight.T.detach().numpy()
        B_tau_matrices = []
        for lag in range(self.max_lags):
            start, end = lag * self.n_features, (lag + 1) * self.n_features
            B_tau = B_lagged_stack[start:end, :].T
            B_tau_matrices.append(B_tau)

        self.adjacency_matrices = [B0_est] + B_tau_matrices
        print(f"Extracted {len(self.adjacency_matrices)} adjacency matrices (B0 to B{self.max_lags}).")

    def rescale_coefficients(self):
        """
        Rescales the learned adjacency matrices back to the original data scale.

        We acknowledge that since we normalize the raw data before training, the
        final learned coefficients do not directly map to the original data's units.
        This normalization is a crucial step because raw data can contain variables
        with vastly different scales (some very large, some very small), which can
        lead to unstable or misleading coefficients in a linear model. Normalization
        ensures a more fair and stable process during model fitting and pruning.

        This function provides a method to recover the coefficients in their original
        scale. This is for anyone who want to interpret the magnitude of causal effects or
        perform downstream tasks like simulating interventions and "what-if" scenarios.
        """
        if self.model is None or not hasattr(self, 'scaler'):
            raise RuntimeError("You must fit the model before rescaling coefficients.")

        # Get the standard deviations for each feature from the saved scaler
        stds = self.scaler.scale_
        
        # Create the scaling ratio matrix: (sigma_j / sigma_i)
        stds_target = stds.reshape(-1, 1) # Column vector for target stds (sigma_j)
        stds_source = stds.reshape(1, -1) # Row vector for source stds (sigma_i)
        scaling_matrix = stds_target / stds_source

        self.rescaled_adjacency_matrices = []
        for b_scaled in self.adjacency_matrices:
            # Apply the formula element-wise: B_orig = B_scaled * (sigma_j / sigma_i)
            b_original = b_scaled * scaling_matrix
            self.rescaled_adjacency_matrices.append(b_original)
            
        print("Coefficients have been rescaled to the original data's units.")
        return self.rescaled_adjacency_matrices