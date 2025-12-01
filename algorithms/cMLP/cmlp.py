"""
Core implementation of MLP-based Neural Granger Causality.

Contains the MLP and cMLP model definitions, training functions (GISTA, Adam),
and helper utilities for regularization.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

# --- Automatic device selection (CPU / GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def activation_helper(activation):
    """Helper to get activation function from string."""
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation is None:
        return lambda x: x
    else:
        raise ValueError(f'unsupported activation: {activation}')


class MLP(nn.Module):
    """
    A single Multi-Layer Perceptron for predicting one time series.
    Uses 1D convolutions to efficiently model lagged dependencies.
    """
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        modules = [nn.Conv1d(num_series, hidden[0], lag)]
        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            modules.append(nn.Conv1d(d_in, d_out, 1))
        self.layers = nn.ModuleList(modules)

        # Move this network to default device (GPU if available)
        self.to(DEVICE)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)
        return X.transpose(2, 1)


class cMLP(nn.Module):
    """
    Component-wise MLP model with one MLP per time series.
    """
    def __init__(self, num_series, lag, hidden, activation='relu'):
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)
        ])

        # Ensure the whole model is on the default device
        self.to(DEVICE)

    def forward(self, X):
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True):
        """
        Extract learned Granger causality.

        Returns:
            torch.Tensor: A (p x p) matrix where GC[i, j]=1 means j -> i.
        """
        GC = [torch.norm(net.layers[0].weight, dim=(0, 2)) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        return GC


# --- Helper and Training Functions ---

def prox_update(network, lam, lr, penalty):
    """Perform in-place proximal update on the first layer weight matrix."""
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0)
            )
    else:
        raise ValueError(f'unsupported penalty: {penalty}')


def regularize(network, lam, penalty):
    """Calculate the regularization term for the first layer weight matrix."""
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError(f'unsupported penalty: {penalty}')


def ridge_regularize(network, lam):
    """Apply ridge penalty to all subsequent layers."""
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    """Move parameter values from best_model to model."""
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def train_model_gista(cmlp, X, lam, lam_ridge, lr, penalty, max_iter,
                      check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=1):
    """
    Train cMLP model with GISTA (Gradient ISTA).
    """
    p = cmlp.p
    lag = cmlp.lag

    # Ensure input data is on the same device as the model
    model_device = next(cmlp.parameters()).device
    X = X.to(model_device)

    cmlp_copy = deepcopy(cmlp)
    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    mse_list, smooth_list, loss_list = [], [], []
    for i in range(p):
        net = cmlp.networks[i]
        mse = loss_fn(net(X[:, :-1]), X[:, lag:, i:i+1])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty)
            loss_list.append(smooth + nonsmooth)

    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean.item()]
    train_mse_list = [mse_mean.item()]
    
    line_search = begin_line_search
    done = [False for _ in range(p)]
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        sum([s for s, d in zip(smooth_list, done) if not d]).backward()

        new_mse_list, new_smooth_list, new_loss_list = [], [], []

        for i in range(p):
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            step = False
            lr_it = lr_list[i]
            net, net_copy = cmlp.networks[i], cmlp_copy.networks[i]

            while not step:
                for param, temp_param in zip(net.parameters(), net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad
                
                prox_update(net_copy, lam, lr_it, penalty)
                
                mse = loss_fn(net_copy(X[:, :-1]), X[:, lag:, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    loss = smooth + regularize(net_copy, lam, penalty)
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((p_ - t_) ** 2) for p_, t_ in zip(net.parameters(), net_copy.parameters())]
                    )
                
                comp = loss_list[i] if monotone else max(last_losses[i])
                if not line_search or (comp - loss) > tol:
                    step = True
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)
                    lr_list[i] = (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay)
                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                else:
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print(f'Network {i + 1} converged')
                        break
            
            net.zero_grad()
            if step:
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

        mse_list, smooth_list, loss_list = new_mse_list, new_smooth_list, new_loss_list

        if sum(done) == p:
            if verbose > 0:
                print(f'All networks converged at iteration {it + 1}')
            break

        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
            train_loss_list.append(loss_mean.item())
            train_mse_list.append(mse_mean.item())

            if verbose > 0:
                print(f"{'-'*10} Iter = {it + 1} {'-'*10}")
                print(f'Total loss = {loss_mean.item():.6f}')
                print(f'Variable usage = {100 * torch.mean(cmlp.GC().float()):.2f}%')

            if not line_search and train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                line_search = True
                if verbose > 0:
                    print('Switching to line search')
                
    return train_loss_list, train_mse_list
