import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PrunedTransformerModel(nn.Module):
    def __init__(self, model_name, pruning_amount=0.2):
        """
        Initializes the Transformer model and applies pruning.

        Args:
            model_name (str): Name of the transformer model to be used.
            pruning_amount (float, optional): The fraction of weights to prune in each Linear layer. Defaults to 0.2.
        """
        super(PrunedTransformerModel, self).__init__()
        self.model_name = model_name
        self.pruning_amount = pruning_amount

        # Load the pre-trained model
        self._load_and_validate_model()

        # Apply pruning
        self._prune_model()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (Tensor): Input IDs to the model.
            attention_mask (Tensor): Attention mask for the input.

        Returns:
            Tensor: Output from the model.
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _load_and_validate_model(self):
        """
        Loads the transformer model and checks for compatibility.
        """
        try:
            self.transformer = AutoModel.from_pretrained(self.model_name)
        except ValueError as e:
            raise ValueError(f"Failed to load model '{self.model_name}'. Ensure it is correctly specified.") from e

    def _prune_model(self):
        """
        Applies pruning to the Linear layers of the model.
        """
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear):
                self._prune_module(module, 'weight', self.pruning_amount)

    def _prune_module(self, module, name, amount):
        """
        Apply pruning to a module.

        Args:
            module (nn.Module): Module to be pruned.
            name (str): Name of the parameter to prune.
            amount (float): Fraction of connections to prune.
        """
        nn.utils.prune.l1_unstructured(module, name=name, amount=amount)
        # Uncomment the line below to make pruning permanent
        # nn.utils.prune.remove(module, name)

if __name__ == "__main__":
    # Example usage
    model = PrunedTransformerModel('bert-base-uncased')  # Replace with the actual model name
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])  # Example input
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])  # Example attention mask
    output = model(input_ids, attention_mask)
    print(output.shape)  # Check the output shape
