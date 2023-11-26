from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

def optimize_hyperparams(train_function, pbounds, init_points=5, n_iter=25, acq='ei', xi=0.01):
    """
    Perform hyperparameter optimization using Bayesian Optimization.

    Args:
        train_function (function): The training function to optimize. 
                                   It should return a metric to maximize.
        pbounds (dict): Dictionary of hyperparameter bounds.
        init_points (int): Number of iterations to randomly explore the hyperparameter space.
        n_iter (int): Number of iterations for bayesian optimization after initial random exploration.
        acq (str): Acquisition function to be used in optimization. Default is 'ei' (Expected Improvement).
        xi (float): Exploration-exploitation trade-off parameter.

    Returns:
        dict: Optimal hyperparameters found.
    """

    optimizer = BayesianOptimization(
        f=train_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
        acq=acq,
        xi=xi
    )

    print(Colours.green(f"Optimal hyperparameters: {optimizer.max['params']}"))
    return optimizer.max['params']

def train_function_example(learning_rate, gamma):
    """
    Example training function that needs to be optimized. 
    Should return a metric to maximize, e.g., validation reward.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for Q-learning.

    Returns:
        float: Metric to maximize, e.g., validation reward.
    """
    # Example implementation (replace with actual training logic)
    # Example: return simulated_validation_reward(learning_rate, gamma)
    return (learning_rate * gamma) / (learning_rate + gamma)  # Replace with real metric

if __name__ == "__main__":
    # Example bounds for hyperparameters
    pbounds = {
        'learning_rate': (1e-5, 1e-2),
        'gamma': (0.85, 0.99)
    }

    # Running the optimization
    optimal_params = optimize_hyperparams(
        train_function=train_function_example,
        pbounds=pbounds,
        init_points=2,
        n_iter=10
    )

    print("Optimized hyperparameters:", optimal_params)
