import torch
from environments.custom_env import CustomEnv  # Replace with your actual environment import
from agents.dql_agent import DoubleQLearningAgentUCB  # Replace with your actual agent import
import argparse

def validate(agent, environment, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, use_exploration=False)  # Turn off exploration
            next_state, reward, done, _ = environment.step(action.item())
            episode_reward += reward
            state = next_state

        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = total_rewards / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def load_agent(filepath, input_dim, hidden_dim, output_dim, n_layers, n_heads, embedding_dim):
    # Load the agent from the given file path. Customize this as per your agent's architecture.
    agent = DoubleQLearningAgentUCB(input_dim, hidden_dim, output_dim, n_layers, n_heads, embedding_dim)
    agent.load_state_dict(torch.load(filepath))
    agent.eval()
    return agent

def main(args):
    # Create the environment
    env = CustomEnv()  # Initialize your environment here

    # Load the trained agent
    agent = load_agent(args.agent_path, args.input_dim, args.hidden_dim, args.output_dim, args.n_layers, args.n_heads, args.embedding_dim)

    # Validate the agent
    validate(agent, env, args.num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained DQN agent")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to the trained agent model")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension of the agent model")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension of the agent model")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension of the agent model")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers in the agent model")
    parser.add_argument("--n_heads", type=int, required=True, help="Number of heads in the agent model")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension in the agent model")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for validation")
    args = parser.parse_args()

    main(args)
