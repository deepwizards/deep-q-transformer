import torch
import torch.nn as nn
from performer_pytorch import Performer  # Import the Performer model

class EfficientTransformerQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, heads, causal=False, ff_mult=4, dropout=0.1):
        """
        Initializes the Efficient Transformer (Performer) Q-Network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of hidden layer in the model.
            output_dim (int): Number of output features (action space).
            depth (int): Number of layers in the Performer.
            heads (int): Number of attention heads in the Performer.
            causal (bool): If True, uses causal attention. Useful for autoregressive tasks.
            ff_mult (int): Factor to multiply with hidden_dim for feedforward layer size. Default is 4.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super(EfficientTransformerQNetwork, self).__init__()
        self.performer = Performer(
            dim = input_dim,
            depth = depth,
            heads = heads,
            causal = causal,
            ff_mult = ff_mult,
            dropout = dropout
        )
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Forward pass through the Efficient Transformer Q-Network.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Output tensor.
        """
        encoded = self.performer(src)
        output = self.fc_out(encoded)
        return output

# Example usage
# q_network = EfficientTransformerQNetwork(input_dim=512, hidden_dim=512, output_dim=10, depth=6, heads=8, causal=False, ff_mult=4, dropout=0.1)
# input_tensor = torch.randn(32, 10, 512)  # Example input tensor (batch_size, seq_len, input_dim)
# output = q_network(input_tensor)
