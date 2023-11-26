import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the positional encoding module.

        Args:
            d_model (int): The dimension of the embeddings (model size).
            max_len (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a matrix that holds the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Calculate the positional encodings once in log space
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for adding positional encodings to input tensor.

        Args:
            x (Tensor): The input tensor to add positional encodings.

        Returns:
            Tensor: The input tensor with added positional encodings.
        """
        x = x + self.pe[:x.size(0), :]
        return x
