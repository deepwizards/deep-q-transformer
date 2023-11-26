# Deep Q-Learning Transformer Project

This project integrates a Transformer model with a Deep Q-Learning algorithm to create a novel language processing model. The implementation is focused on harnessing the strengths of both architectures for improved performance in language-related tasks.

## Integration of Q-Learning and Transformers for Text Generation

```
+---------------------+      +----------------------+
|                     |      |                      |
|  PrunedTransformer  |----->|  DeepQLearningModel  |
|    Model (NLP)      |      |    (DQN)             |
|                     |      |                      |
+---------------------+      +----------------------+
       ^                                |
       |                                |
       |         +------------------+   |
       |         |                  |   |
       +---------|  ExperienceReplay|<--+
                 |     Buffer       |
                 |                  |
                 +------------------+
                        ^
                        |
+-----------------------+------------------------+
|                                                |
|                  DataLoader                    |
|                                                |
+------------------------------------------------+

```

### Overview
This implementation uniquely combines the concepts of Q-Learning and Deep Q-Networks (DQN) with Transformer models, aiming to create a robust system for text generation. Here's how the various components work together:

### Deep Q-Learning in Text Generation
- **DeepQLearningModel**: Represents a neural network to approximate the Q-function, crucial in Q-learning. It predicts the value of taking certain actions based on state representations provided by the Transformer model.
- **Custom Loss Function**: This function calculates the loss for updating Q-values, adhering to the temporal-difference learning approach of Q-learning. It factors in rewards and discounted future rewards, aligning with the objective to maximize expected rewards.

### Integration with Transformer Models
- **PrunedTransformerModel**: Generates embeddings or state representations of input text. This state is essential for the DQN, which requires a state input in the context of text generation.
- **State and Action Interplay**: The output of the Transformer model (state embeddings) is fed into the DeepQLearningModel. This model then predicts action values (Q-values) for each possible action, such as the next word in a text sequence.

### Experience Replay
- **ExperienceReplay Class**: Implements the concept of experience replay, a critical aspect for stabilizing DQN training. By storing and randomly sampling past experiences, it breaks correlations in sequential data, enhancing learning stability.
- **Significance in Text Generation**: This approach is vital in text generation where sequential dependencies can lead to unstable learning dynamics if not properly managed.

### Combining for Text Generation
- **Training Loop (train_model)**: This function unites the Transformer model with the DQN. Designed for scenarios where actions (like choosing the next word) are part of text generation, the DQN is trained to optimize these actions based on Transformer-provided state representations.
- **Hybrid Model Functionality**: Utilizes Transformers for understanding and processing text (state) and DQN for learning the policy of generating text (actions) in this context.

### Conclusion
The implementation is an innovative fusion of Transformers and Deep Q-Learning, tailored for text generation. It leverages the strengths of Transformers in text processing and understanding, alongside DQN for decision-making in text generation, such as next-word prediction. The inclusion of experience replay and a custom loss function for the DQN aligns with the principles of stabilizing and optimizing Q-learning in complex domains like text generation.

## Project Structure

- `src/`
  - `transformer_model.py`: Contains the `PrunedTransformerModel` class, a pruned version of a Transformer model.
  - `dql_model.py`: Includes the `DeepQLearningModel` and `ExperienceReplay` classes for the Deep Q-Learning algorithm.
  - `training_loop.py`: Implements the training loop for the integrated model.
- `main.py`: The main script to run the project, setting up the models, loading data, and initiating training.
- `README.md`: This file, providing an overview and instructions for the project.

## Installation

1. Clone the repository:

`git clone https://github.com/deepwizards/deep-q-transformer.git`

2. Install dependencies:

`pip install -r requirements.txt`

## Usage

To run the project, execute the `main.py` script:

`python main.py`

## Testing

This project includes a comprehensive suite of unit tests to ensure the reliability and correctness of its components. The tests cover models, utility functions, and training procedures. 

### Key Components Tested
- `PrunedTransformerModel`: Verifies the functionality and pruning feature of the Transformer model.
- `DeepQLearningModel`: Ensures the forward pass and output range of the Deep Q-Learning model.
- `ExperienceReplay`: Checks the behavior of the Experience Replay Buffer, including buffer overflow handling.
- `custom_loss`: Validates the custom loss function specific to our Deep Q-Learning setup.
- `train_model`: Tests the training loop, including data loading and model updating processes.

`python -m unittest`


This command will automatically discover and run all tests written in files following the `test_*.py` naming pattern.

### Test Structure
Each test class in our test suite corresponds to a specific component or functionality in the project. For instance:

- `TestPrunedTransformerModel` contains tests to ensure that the Transformer model is initialized correctly and prunes its weights as expected.
- `TestDeepQLearningModel` checks if the DQL model produces outputs of the correct shape and value range.
- `TestExperienceReplay` ensures that the replay buffer operates correctly under various scenarios, like adding new experiences and handling capacity limits.
- `TestCustomLoss` focuses on the behavior of the custom loss function under different input scenarios.
- `TestTrainModel` verifies the overall training loop, including interactions between different components like the model, data loader, and optimizer.

### Adding New Tests
When extending the project with new features or components, corresponding tests should be added following the existing structure. This practice ensures that new additions maintain the overall integrity and functionality of the project.

For writing new tests, follow the Python `unittest` framework's guidelines and ensure that each test is focused, independent, and covers both typical use cases and edge cases.

### Test Mocking
Some tests use mocking (via `unittest.mock`) to isolate the tested functionality and speed up the testing process by avoiding dependencies on external resources like file systems or networks.

### Running the Tests
To run the tests, you need to have `unittest`, a built-in Python testing framework, and other project dependencies installed. You can execute the tests by running the following command in your project's root directory:

## Implementation Details

### PrunedTransformerModel

- A Transformer model (e.g., BERT, LLaMA) with pruning applied to its linear layers.
- Usage:
`model = PrunedTransformerModel('bert-base-uncased')`

### DeepQLearningModel

- A neural network model implementing Deep Q-Learning.
- Usage:
`dq_model = DeepQLearningModel(input_dim=768, action_dim=10)`

### ExperienceReplay

- A class for implementing experience replay in reinforcement learning.
- Usage:
`replay_buffer = ExperienceReplay(capacity=10000)`

### Training Loop

- Located in `training_loop.py`, it integrates the training process for both Transformer and DQL models.
- Usage:
`train_model(transformer, dq_learning_model, optimizer, epochs, tokenizer, data_loader, replay_buffer, device)`

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
