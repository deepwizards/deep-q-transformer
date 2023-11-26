import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from src.transformer_model import PrunedTransformerModel
from src.dql_model import EnhancedDQLModel, ExperienceReplay
from src.training_loop import train_model
import os

class StreamTextDataset(IterableDataset):
    """
    A custom IterableDataset for streaming text data from files.
    """
    def __init__(self, directory, tokenizer, max_length, action_space_size):
        """
        Initializes the dataset with directory and tokenizer settings.

        Args:
            directory (str): Path to the directory containing text files.
            tokenizer (AutoTokenizer): Tokenizer for text processing.
            max_length (int): Maximum sequence length for tokenization.
            action_space_size (int): Size of the action space.
        """
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.action_space_size = action_space_size

    def process_file(self, file_path):
        """
        Process a single file and yield tokenized inputs and targets.

        Args:
            file_path (str): Path to the text file.

        Yields:
            tuple: Tuple containing input_ids, attention_mask, action, and reward.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokenized_text = self.tokenizer.tokenize(text)

        for i in range(0, len(tokenized_text) - 1):
            input_sequence = ' '.join(tokenized_text[max(0, i - self.max_length + 1):i + 1])
            target_sequence = tokenized_text[i + 1]

            input_tokens = self.tokenizer(input_sequence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            target_tokens = self.tokenizer(target_sequence, return_tensors="pt")

            input_ids = input_tokens['input_ids'].squeeze(0)
            attention_mask = input_tokens['attention_mask'].squeeze(0)
            action = torch.argmax(input_tokens['input_ids'], dim=-1)
            reward = torch.tensor(1.0 if action[-1] == target_tokens['input_ids'][0, 0] else 0.0)

            yield input_ids, attention_mask, action, reward

    def __iter__(self):
        """
        Iterator over the files in the directory.

        Yields:
            tuple: Tuple containing input_ids, attention_mask, action, and reward from each file.
        """
        for filename in os.listdir(self.directory):
            file_path = os.path.join(self.directory, filename)
            yield from self.process_file(file_path)

def load_data(directory, tokenizer, batch_size=16, max_length=512, action_space_size=10):
    """
    Loads data from the given directory using StreamTextDataset.

    Args:
        directory (str): Directory containing the text files.
        tokenizer (AutoTokenizer): Tokenizer for text processing.
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum sequence length for tokenization.
        action_space_size (int): Size of the action space.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    dataset = StreamTextDataset(directory, tokenizer, max_length, action_space_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def main():
    """
    Main function to initialize models, load data, and start training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        transformer_model = PrunedTransformerModel('bert-base-uncased').to(device)
        dq_learning_model = EnhancedDQLModel(input_dim=768, output_dim=10).to(device)
    except Exception as e:
        print(f"Error in model initialization: {e}")
        return

    optimizer = torch.optim.Adam([
        {'params': transformer_model.parameters(), 'lr': 1e-5},
        {'params': dq_learning_model.parameters(), 'lr': 1e-4}
    ])

    replay_buffer = ExperienceReplay(capacity=10000)
    directory = 'data/train'  # Update with the actual directory path

    try:
        data_loader = load_data(directory, tokenizer)
    except Exception as e:
        print(f"Error in data loading: {e}")
        return

    epochs = 10
    train_model(transformer_model, dq_learning_model, optimizer, epochs, tokenizer, data_loader, replay_buffer, device)

if __name__ == "__main__":
    main()
