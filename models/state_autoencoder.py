import torch
import torch.nn as nn

class StateAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        """
        Initializes the StateAutoEncoder.

        Args:
            input_dim (int): The dimensionality of the input state.
            embedding_dim (int): The dimensionality of the embedded state.
        """
        super(StateAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): The input tensor representing the state.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the embedded state and the reconstructed state.
        """
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)
        return embedded, reconstructed
