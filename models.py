import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int, vocab_size : int, dropout : float, device : torch.device):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device

        self.embeddings = nn.Embedding(embedding_dim=self.input_size, num_embeddings=self.vocab_size).to(self.device)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout if self.num_layers > 2 else 0.0, device=self.device)
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

    def forward(self, x, hc_0 = None):
        # Pass the input through the embedding layer
        x = x.to(self.device)
        x = self.embeddings(x)
        if hc_0 is not None:
            (h_0, c_0) = hc_0
            expected_shape = (self.num_layers, x.size(0), self.hidden_size)
            if h_0.shape != expected_shape:
                # print(f"Hidden state mismatch ! Current shape: {h_0.shape} Expected: {expected_shape}")
                h_0, c_0 = self.init_hidden(x.size(0))

            out, state = self.lstm(x, (h_0, c_0))
        else:
            out, state = self.lstm(x)

        out = self.output_layer(out[:, -1, :])
        out = out.unsqueeze(dim=1)
        return out, state

