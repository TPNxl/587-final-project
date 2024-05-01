import torch
import torch.nn as nn
import numpy as np
import math
from models.GInv_Linear import *
from models.GInv_structures import subspace

class GInvariantLSTMLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.i_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.f_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.g_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)
        self.o_inv_layer = GInvariantLayer(input_dim=input_dim, output_dim=hidden_dim, device=self.device)

        self.i_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device)
        self.f_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device)
        self.g_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device)
        self.o_lin_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, device=self.device)

        self.i_act_func = torch.nn.Sigmoid()
        self.f_act_func = torch.nn.Sigmoid()
        self.g_act_func = torch.nn.Sigmoid()
        self.o_act_func = torch.nn.Sigmoid()
        self.c_act_func = torch.nn.Sigmoid()
        
    def to(self, device):
        self.i_inv_layer = self.i_inv_layer.to(device)
        self.f_inv_layer = self.f_inv_layer.to(device)
        self.g_inv_layer = self.g_inv_layer.to(device)
        self.o_inv_layer = self.o_inv_layer.to(device)
        self.i_lin_layer = self.i_lin_layer.to(device)
        self.f_lin_layer = self.f_lin_layer.to(device)
        self.g_lin_layer = self.g_lin_layer.to(device)
        self.o_lin_layer = self.o_lin_layer.to(device)
        
    def forward(self, X, H=None):
        '''
        Forward pass for the GInvariantLSTM.
        
        Inputs: 
        X: Input tensor of size (N, L, input_dim)
        H: Initial hidden state of size (N, hidden_dim)
        
        Returns:
        out: Output tensor of size (N, L, hidden_dim)'''
        if X.device != self.device:
            X = X.to(self.device)
        if X.dtype != torch.float:
            print("X is not float")
            X = X.float()
        if H == None:
            H = torch.zeros(X.shape[0], self.hidden_dim, requires_grad=False, device=self.device)
        elif H.dtype != torch.float:
            print("H is not float")
            H = H.float()
        if H.device != self.device:
            H = H.to(self.device)

        H.detach_()

        #self.hidden_state = H
        #self.hidden_state.detach_()

        X = torch.swapdims(X, 0, 1)

        i = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        f = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        g = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        o = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        cell = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        output = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        for iter in range(X.shape[0]):
            if iter == 0:
                i[iter] = self.i_act_func(self.i_inv_layer.forward(X[iter]) + self.i_lin_layer.forward(H))
                f[iter] = self.f_act_func(self.f_inv_layer.forward(X[iter]) + self.f_lin_layer.forward(H))
                g[iter] = self.g_act_func(self.g_inv_layer.forward(X[iter]) + self.g_lin_layer.forward(H))
                o[iter] = self.o_act_func(self.o_inv_layer.forward(X[iter]) + self.o_lin_layer.forward(H))
                cell[iter] = torch.mul(i[iter], g[iter])
            else:
                i[iter] = self.i_act_func(self.i_inv_layer.forward(X[iter]) + self.i_lin_layer.forward(output[iter - 1]))
                f[iter] = self.f_act_func(self.f_inv_layer.forward(X[iter]) + self.f_lin_layer.forward(output[iter - 1]))
                g[iter] = self.g_act_func(self.g_inv_layer.forward(X[iter]) + self.g_lin_layer.forward(output[iter - 1]))
                o[iter] = self.o_act_func(self.o_inv_layer.forward(X[iter]) + self.o_lin_layer.forward(output[iter - 1]))
                cell[iter] = torch.mul(f[iter], cell[iter - 1]) + torch.mul(i[iter], g[iter])
            output[iter] = torch.mul(o[iter], self.c_act_func(cell[iter].clone()))

        # output = torch.empty((X.shape[0], X.shape[1], self.hidden_dim), device=self.device)
        # for iter in range(X.shape[0]):
        #     if iter == 0:
        #         i[iter] = self.i_act_func(self.i_inv_layer.forward(X[iter]) + self.i_lin_layer.forward(H))
        #         f[iter] = self.f_act_func(self.f_inv_layer.forward(X[iter]) + self.f_lin_layer.forward(H))
        #         g[iter] = self.g_act_func(self.g_inv_layer.forward(X[iter]) + self.g_lin_layer.forward(H))
        #         o[iter] = self.o_act_func(self.o_inv_layer.forward(X[iter]) + self.o_lin_layer.forward(H))
        #         cell[iter] = torch.mul(i[iter], g[iter])
        #     else:
        #         i[iter] = self.i_act_func(self.i_inv_layer.forward(X[iter]) + self.i_lin_layer.forward(output[iter - 1]))
        #         f[iter] = self.f_act_func(self.f_inv_layer.forward(X[iter]) + self.f_lin_layer.forward(output[iter - 1]))
        #         g[iter] = self.g_act_func(self.g_inv_layer.forward(X[iter]) + self.g_lin_layer.forward(output[iter - 1]))
        #         o[iter] = self.o_act_func(self.o_inv_layer.forward(X[iter]) + self.o_lin_layer.forward(output[iter - 1]))
        #         cell[iter] = torch.mul(f[iter], cell[iter - 1]) + torch.mul(i[iter], g[iter])
        #     output[iter] = torch.mul(o[iter], self.c_act_func(cell[iter]))

        return torch.swapdims(output, 0, 1)

class GInvariantLSTM_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation=None, out_size=1, num_layers=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        R'''
        Init for a G-invariant LSTM. Modeled after Torch documentation except with an invariant MLP for the new input layer.
        Assume batch_first = True.'''
        super().__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.device = device
        self.out_dim = out_size

        self.first_layer = GInvariantLSTMLayer(input_size, hidden_size, device=self.device)
        self.LSTM = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=out_size),
        )

        self.first_layer.to(device)
        self.LSTM = self.LSTM.to(device)
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
            print("Tensor is not float")
            x = x.float()
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        if h_0 is None:
            h_0 = torch.zeros((x.shape[0], self.hidden_dim), requires_grad=False, dtype=x.dtype, device=self.device)
        elif h_0.device != self.device:
            h_0 = h_0.to(self.device)
        
        # print(x.shape)
        GInv_LSTM_out = self.first_layer.forward(x, h_0)
        # print(GInv_LSTM_out.shape)
        # print(GInv_LSTM_out.device)
        # print(GInv_LSTM_out.type())
        LSTM_out, _ = self.LSTM(GInv_LSTM_out)
        
        reshaped = torch.reshape(LSTM_out, (x.shape[0] * x.shape[1], -1))
        # print(reshaped.shape)
        out = self.MLP(reshaped)
        # print(out.shape)

        return torch.reshape(out, (x.shape[0], x.shape[1], self.out_dim))


