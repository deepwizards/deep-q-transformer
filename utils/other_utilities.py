import torch
import os
import json
import numpy as np
import logging
from datetime import datetime

def save_model(model, path, filename):
    """
    Save a PyTorch model state to a file.

    Args:
        model (torch.nn.Module): The model to be saved.
        path (str): The directory path to save the model file.
        filename (str): The filename for the saved model.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))

def load_model(model, path, filename):
    """
    Load a PyTorch model state from a file.

    Args:
        model (torch.nn.Module): The model to load state into.
        path (str): The directory path where the model file is located.
        filename (str): The filename of the saved model.
    """
    model.load_state_dict(torch.load(os.path.join(path, filename)))

def setup_logging(log_dir='logs', log_filename='training.log'):
    """
    Set up logging for the project.

    Args:
        log_dir (str): Directory to store log files.
        log_filename (str): Filename for the log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename)),
            logging.StreamHandler()
        ]
    )

def log_info(message):
    """
    Log an informational message.

    Args:
        message (str): The message to log.
    """
    logging.info(message)

def save_hyperparameters(params, path, filename='hyperparameters.json'):
    """
    Save hyperparameters to a JSON file.

    Args:
        params (dict): Hyperparameters to save.
        path (str): The directory path to save the file.
        filename (str): The filename for the saved parameters.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, filename), 'w') as file:
        json.dump(params, file, indent=4)

def load_hyperparameters(path, filename='hyperparameters.json'):
    """
    Load hyperparameters from a JSON file.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The filename of the parameters file.

    Returns:
        dict: The loaded hyperparameters.
    """
    with open(os.path.join(path, filename), 'r') as file:
        return json.load(file)

def normalize_data(data, mean=None, std=None):
    """
    Normalize data.

    Args:
        data (np.array): The data to normalize.
        mean (float, optional): The mean value for normalization. If None, calculate from data.
        std (float, optional): The standard deviation for normalization. If None, calculate from data.

    Returns:
        np.array: Normalized data.
    """
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std

# Additional utility functions can be added as needed.
