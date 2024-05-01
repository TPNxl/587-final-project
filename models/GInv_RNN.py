import torch
import torch.nn as nn
import numpy as np
import math
from models.GInv_Linear import *
from models.GInv_structures import subspace

class GInvariantRecurrentLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant RNN. Modeled after Torch documentation except with an invariant MLP for the new input layer.
        Assume batch_first = True.'''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.hidden_state = None

        self.inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)

        self.hidden_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device)

        if activation is None:
            self.activation = torch.nn.ReLU()
        else:
            self.activation = activation
        

    def forward(self, x, h_0=None):
        R'''
        Forward pass for the GInvariantRNN.
        
        Inputs: 
        x: Input tensor of size (N, L, input_dim)
        h_0: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (N, L, hidden_dim)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.float:
            print("Tensor is not float")
            x = x.float()
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), requires_grad=False, dtype=torch.float, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        self.hidden_state = h_0

        x = torch.swapdims(x, 0, 1)

        out = torch.empty((x.shape[0], x.shape[1], self.hidden_dim), dtype=torch.float, device=self.device)

        for i in range(x.shape[0]):
            hidden_state = self.inv_layer.forward(x[i]) + self.hidden_layer.forward(self.hidden_state)
            hidden_state = self.activation(hidden_state)
            out[i] = hidden_state
            self.hidden_state = hidden_state
        
        return torch.swapdims(out, 0, 1)

class GInvariantRNN_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, activation=None, num_layers=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant RNN. Modeled after Torch documentation except with an invariant MLP for the new input layer.'''
        super().__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.device = device
        self.out_size = out_size

        self.first_layer = GInvariantRecurrentLayer(input_size, hidden_size, activation, device=device)
        self.dropout_func = None
        self.RNN = torch.nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=out_size),
        )

        self.first_layer.to(device)
        self.RNN = self.RNN.to(device)
        self.MLP = self.MLP.to(device)
        

    def forward(self, x, h_0=None):
        R'''
        Forward pass for the GInvariantRNN.
        
        Inputs: 
        x: Input tensor of size (L, N, input_dim)
        h_0: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (L, N, hidden_dim)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.float:
            x = x.float()
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), dtype=torch.float, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        # print(x.shape)
        GInv_RNN_out = self.first_layer.forward(x, h_0)
        # print(GInv_RNN_out.shape)
        #print(GInv_RNN_out.device)
         #print(GInv_RNN_out.type())
        RNN_out, _ = self.RNN(GInv_RNN_out)
        
        reshaped = torch.reshape(RNN_out, (x.shape[0] * x.shape[1], -1))
        # print(reshaped.shape)
        out = self.MLP(reshaped)
        # print(out.shape)
        out = torch.reshape(out, (x.shape[0], x.shape[1], self.out_size))
        return out
