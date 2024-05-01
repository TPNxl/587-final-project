import torch

class MLP_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        '''
        Init for an LSTM model. Modeled after Torch documentation.
        Assume batch_first = True.'''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.dropout = dropout
        self.device = device

        self.first_layer = torch.nn.Linear(in_features=input_size, out_features=hidden_size)

        self.MLP = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=out_size),
        )

        self.first_layer = self.first_layer.to(device)
        self.MLP = self.MLP.to(device)

    def forward(self, x: torch.Tensor):
        R'''
        Forward pass for the LSTM.
        
        Inputs: 
        x: Input tensor of size (N, L, input_dim)
        
        Returns:
        out: Output tensor of size (N, L, out_size)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != torch.float:
            # print("Tensor is not float")
            x = x.float()
        
        reshaped = torch.reshape(x, (x.shape[0] * x.shape[1], -1))
        # print(x.type())
        # print(reshaped.shape)
        first_out = self.first_layer(reshaped)
        # print(first_out.shape)
        out = self.MLP(first_out)
        # print(out.shape)
        out = torch.reshape(out, (x.shape[0], x.shape[1], self.out_size))
        return out


        
