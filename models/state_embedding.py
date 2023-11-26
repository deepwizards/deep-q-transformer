import torch
import torch.nn as nn

class StateEmbedding(nn.Module):
    """
    State Embedding class for transforming input states into embeddings.

    Attributes:
        embedding (nn.Linear): Linear layer to transform input state to embedding.

    Args:
        input_dim (int): Dimension of the input state.
        embedding_dim (int): Dimension of the output embedding.
    """

    def __init__(self, input_dim, embedding_dim):
        """
        Initializes the StateEmbedding module.

        Args:
            input_dim (int): The size of the input state vector.
            embedding_dim (int): The size of the embedding vector.
        """
        super(StateEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the StateEmbedding module.

        Args:
            x (Tensor): The input tensor representing the state.

        Returns:
            Tensor: An embedded representation of the input state.
        """
        return self.embedding(x)
