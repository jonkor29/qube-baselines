#!/usr/bin/env python3

import numpy as np
import gym
import os
import argparse

# New imports for simulator parameter handling
from scipy.stats import multivariate_normal
from load_config import load_config, params_from_config_dict

from gym_brt.envs import (
    QubeSwingupEnv,
    QubeSwingupSparseEnv,
    QubeSwingupFollowEnv,
    QubeSwingupFollowSparseEnv,
    QubeBalanceEnv,
    QubeBalanceSparseEnv,
    QubeBalanceFollowEnv,
    QubeBalanceFollowSparseEnv,
    QubeDampenEnv,
    QubeDampenSparseEnv,
    QubeDampenFollowEnv,
    QubeDampenFollowSparseEnv,
    QubeRotorEnv,
    QubeRotorFollowEnv,
)

from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2


def evaluate_model(
    env_name,
    model_path,
    num_episodes=10,
    use_hardware=False,
    render=False,
    deterministic=True,
    sim_params=None,
    deterministic_resets=True,
):
    """
    Evaluate a trained model.

    env_name: (str) the name of the environment to run
    model_path: (str) path to the trained model
    num_episodes: (int) number of episodes to run
    use_hardware: (bool) whether to use the hardware or the simulator
    render: (bool) whether to render the environment
    deterministic: (bool) whether to use a deterministic policy
    sim_params: (list or None) A list of physical parameters for the simulator.
    deterministic_resets: (bool) Whether to use deterministic or stochastic resets in the simulator.
    """
    envs = {
        "QubeSwingupEnv": QubeSwingupEnv,
        "QubeSwingupSparseEnv": QubeSwingupSparseEnv,
        "QubeSwingupFollowEnv": QubeSwingupFollowEnv,
        "QubeSwingupFollowSparseEnv": QubeSwingupFollowSparseEnv,
        "QubeBalanceEnv": QubeBalanceEnv,
        "QubeBalanceSparseEnv": QubeBalanceSparseEnv,
        "QubeBalanceFollowEnv": QubeBalanceFollowEnv,
        "QubeBalanceFollowSparseEnv": QubeBalanceFollowSparseEnv,
        "QubeDampenEnv": QubeDampenEnv,
        "QubeDampenSparseEnv": QubeDampenSparseEnv,
        "QubeDampenFollowEnv": QubeDampenFollowEnv,
        "QubeDampenFollowSparseEnv": QubeDampenFollowSparseEnv,
        "QubeRotorEnv": QubeRotorEnv,
        "QubeRotorFollowEnv": QubeRotorFollowEnv,
    }

    def make_env():
        p_phi = None
        domain_rand = False
        if not use_hardware and sim_params is not None:
            # Create a distribution with zero variance to use the exact specified parameters
            p_phi = multivariate_normal(
                mean=sim_params,
                cov=np.diag(np.zeros(len(sim_params))),
                allow_singular=True,
            )
            # domain_randomization must be True for the environment to use p_phi
            domain_rand = True

        env_out = envs[env_name](
            use_simulator=not use_hardware,
            frequency=250,
            p_phi=p_phi,
            domain_randomization=domain_rand,
            deterministic_resets=deterministic_resets
        )
        return env_out

    env = DummyVecEnv([make_env])
    model = PPO2.load(model_path, env=env)

    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0] # Reward is a vector with one element
            if render and not use_hardware:
                env.render()
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\nEvaluation results for {model_path}:")
    print(f"Mean reward over {num_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save rewards to a file in the model's directory
    reward_file_path = os.path.join(os.path.dirname(model_path), "reward.txt")
    with open(reward_file_path, "w") as f:
        f.write(f"real_rollouts_rewards: {episode_rewards}\n")
        f.write(f"mean_reward: {mean_reward}\n")
        f.write(f"std_reward: {std_reward}\n")
    print(f"Reward data saved to {reward_file_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the Qube environment.")
    parser.add_argument(
        "-e", "--env", type=str, required=True, help="Environment name"
    )
    parser.add_argument(
        "-l", "--load", type=str, required=True, help="Path to the trained model (.pkl file)"
    )
    parser.add_argument(
        "-ne", "--num-episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument("-hw", "--use-hardware", action="store_true", help="Use the physical Qube hardware instead of the simulator.")
    parser.add_argument("-r", "--render", action="store_true", help="Render the environment (only for simulator).")
    parser.add_argument("--non-deterministic", action="store_true", help="Use a stochastic policy for evaluation.")
    
    parser.add_argument(
        "--sim-params",
        type=float,
        nargs='+',
        default=None,
        help="Simulator only: A list of physical parameters to use, overriding the config file. "
             "Order: [Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g]"
             "Config (for reference): 7.5 0.042 0.042 0.095 0.085 0.00027 0.024 0.129 0.00005 9.81",
    )
    parser.add_argument(
        "--deterministic-resets",
        action="store_true",
        help="Simulator only: Use deterministic environment resets instead of stochastic ones.",
    )
    args = parser.parse_args()

    sim_params = args.sim_params
    if not args.use_hardware and sim_params is None:
        print("Loading default simulator parameters from config.yaml...")
        # Load params from config.yaml based on QUANSER_HW env var or default
        try:
            config_dict = load_config() 
            sim_params = params_from_config_dict(config_dict)
            print(f"Using default parameters: {np.array2string(sim_params, precision=4, suppress_small=True)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load default config. {e}. Simulator will use its internal defaults.")

    if args.use_hardware:
        sim_params = None

    evaluate_model(
        env_name=args.env,
        model_path=args.load,
        num_episodes=args.num_episodes,
        use_hardware=args.use_hardware,
        render=args.render,
        deterministic=not args.non_deterministic,
        sim_params=sim_params,
        deterministic_resets= args.deterministic_resets,
    )
