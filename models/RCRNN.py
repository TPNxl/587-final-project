import torch

class RCRNN_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layer_sizes: list, input_len=600, output_len=30, out_size=1, dropout=0.0, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        '''
        Init for an RNN -> CNN -> RNN (RCRNN) model.
        Assume batch_first = True.'''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.device = device

        self.layer_list = []
        self.layer_list.append(torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, batch_first=True))
        layer_sizes.insert(0, input_len)
        layer_sizes.append(output_len)
        for i in range(len(layer_sizes) - 1):
            self.layer_list.append(torch.nn.Conv2d(1, 1, (layer_sizes[i] - layer_sizes[i+1] + 1, 1), padding="same",dtype=torch.float64))
            self.layer_list.append(torch.nn.BatchNorm2d(1))
            self.layer_list.append(torch.nn.ReLU())
            self.layer_list.append(torch.nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, dropout=dropout, batch_first=True))
        
        self.MLP = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size), 
                          torch.nn.ReLU(),
                          torch.nn.Linear(in_features=hidden_size, out_features=out_size))

        for layer in self.layer_list:
            layer = layer.to(self.device)
   
        

    def forward(self, x: torch.Tensor):
        R'''
        Forward pass for the RCRNN.
        
        Inputs: 
        x: Input tensor of size (N, L, input_dim)
        
        Returns:
        out: Output tensor of size (N, L, out_size)'''

        if x.device != self.device:
            x = x.to(self.device)
        if x.type != torch.float32:
            x = x.type(torch.float32)
        
        out = x
        for layer in self.layer_list:
            out = layer(out)
        reshaped = torch.reshape(out, (x.shape[0] * x.shape[1], -1))
        out = self.MLP(reshaped)
        out = torch.reshape(out, (x.shape[0], x.shape[1], self.out_size))
        return out


        
