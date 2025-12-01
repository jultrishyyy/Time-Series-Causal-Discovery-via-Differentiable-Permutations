"""
Core implementation of LSTM-based Neural Granger Causality.

Contains the LSTM and cLSTM model definitions, training functions (GISTA, Adam),
and helper utilities for regularization and data preparation.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

# --- Automatic device selection (CPU / GPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """
    LSTM model with an output layer to generate predictions for a single series.
    """
    def __init__(self, num_series, hidden):
        super(LSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden
        self.lstm = nn.LSTM(num_series, hidden, batch_first=True)
        self.lstm.flatten_parameters()
        self.linear = nn.Conv1d(hidden, 1, 1)

        # Move model to the default device (GPU if available, otherwise CPU)
        self.to(DEVICE)

    def init_hidden(self, batch):
        """Initialize hidden states for LSTM cell."""
        device = self.lstm.weight_ih_l0.device
        return (torch.zeros(1, batch, self.hidden, device=device),
                torch.zeros(1, batch, self.hidden, device=device))

    def forward(self, X, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(X.shape[0])
        X, hidden = self.lstm(X, hidden)
        X = X.transpose(2, 1)
        X = self.linear(X)
        return X.transpose(2, 1), hidden


class cLSTM(nn.Module):
    """
    cLSTM model with one LSTM per time series.
    """
    def __init__(self, num_series, hidden):
        super(cLSTM, self).__init__()
        self.p = num_series
        self.hidden = hidden
        self.networks = nn.ModuleList([
            LSTM(num_series, hidden) for _ in range(num_series)
        ])

        # Ensure the whole container module is on the default device as well
        self.to(DEVICE)

    def forward(self, X, hidden=None):
        if hidden is None:
            hidden = [None for _ in range(self.p)]
        pred = [self.networks[i](X, hidden[i]) for i in range(self.p)]
        pred, hidden = zip(*pred)
        pred = torch.cat(pred, dim=2)
        return pred, hidden

    def GC(self, threshold=True):
        """
        Extract learned Granger causality.

        Args:
            threshold (bool): If True, returns a binary matrix (0 or 1). 
                              If False, returns the norm of weights.
        
        Returns:
            torch.Tensor: A (p x p) matrix where GC[i, j]=1 means j -> i.
        """
        GC = [torch.norm(net.lstm.weight_ih_l0, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


# --- Helper and Training Functions ---

def prox_update(network, lam, lr):
    """Perform in-place proximal update on first layer weight matrix."""
    W = network.lstm.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr))) *
              torch.clamp(norm - (lr * lam), min=0.0))
    network.lstm.flatten_parameters()


def regularize(network, lam):
    """Calculate group sparsity regularization term."""
    W = network.lstm.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))


def ridge_regularize(network, lam):
    """Apply ridge penalty at linear layer and hidden-hidden weights."""
    return lam * (torch.sum(network.linear.weight ** 2) +
                  torch.sum(network.lstm.weight_hh_l0 ** 2))


def restore_parameters(model, best_model):
    """Move parameter values from best_model to model."""
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def arrange_input(data, context):
    """
    Arrange a single time series into overlapping short sequences.
    """
    assert context >= 1 and isinstance(context, int)
    input_data = torch.zeros(len(data) - context, context, data.shape[1],
                             dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input_data[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input_data.detach(), target.detach()


def train_model_gista(clstm, X, context, lam, lam_ridge, lr, max_iter,
                      check_every=50, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=1):
    """
    Train cLSTM model with GISTA (Gradient ISTA).
    """
    p = clstm.p

    # Ensure data is on the same device as the model (automatic CPU/GPU handling)
    model_device = next(clstm.parameters()).device
    X = [x.to(model_device) for x in X]

    clstm_copy = deepcopy(clstm)
    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)]

    # Set up data.
    X, Y = zip(*[arrange_input(x, context) for x in X])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)

    # Calculate full loss.
    mse_list, smooth_list, loss_list = [], [], []
    for i in range(p):
        net = clstm.networks[i]
        pred, _ = net(X)
        mse = loss_fn(pred[:, :, 0], Y[:, :, i])
        ridge = ridge_regularize(net, lam_ridge)
        smooth = mse + ridge
        mse_list.append(mse)
        smooth_list.append(smooth)
        with torch.no_grad():
            nonsmooth = regularize(net, lam)
            loss_list.append(smooth + nonsmooth)

    # Set up lists for loss and mse.
    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]
    
    line_search = begin_line_search
    done = [False for _ in range(p)]
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    for it in range(max_iter):
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        new_mse_list, new_smooth_list, new_loss_list = [], [], []

        for i in range(p):
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            step = False
            lr_it = lr_list[i]
            net = clstm.networks[i]
            net_copy = clstm_copy.networks[i]

            while not step:
                for param, temp_param in zip(net.parameters(), net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad
                
                prox_update(net_copy, lam, lr_it)

                pred, _ = net_copy(X)
                mse = loss_fn(pred[:, :, 0], Y[:, :, i])
                ridge = ridge_regularize(net_copy, lam_ridge)
                smooth = mse + ridge
                with torch.no_grad():
                    loss = smooth + regularize(net_copy, lam)
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((p_ - t_) ** 2) for p_, t_ in zip(net.parameters(), net_copy.parameters())])
                
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
                clstm.networks[i], clstm_copy.networks[i] = net_copy, net

        mse_list, smooth_list, loss_list = new_mse_list, new_smooth_list, new_loss_list

        if sum(done) == p:
            if verbose > 0:
                print(f'Done at iteration = {it + 1}')
            break

        if (it + 1) % check_every == 0:
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)

            if verbose > 0:
                print(f"{'-'*10} Iter = {it + 1} {'-'*10}")
                print(f'Total loss = {loss_mean.item():.6f}')
                print(f'Variable usage = {100 * torch.mean(clstm.GC().float()):.2f}%')

            if not line_search and train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                line_search = True
                if verbose > 0:
                    print('Switching to line search')

    return train_loss_list, train_mse_list
