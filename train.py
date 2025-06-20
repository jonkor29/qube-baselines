#!/usr/bin/env python3

import numpy as np
import gym
import os
import json
from load_config import load_config
from datetime import datetime 

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

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import bench, logger

from stable_baselines.ppo2 import PPO2


def init_save_callback(logdir, batch_size, save_interval):
    def callback(
        _locals,
        _globals,
        logdir=logdir,
        batch_size=batch_size,
        save_interval=save_interval,
    ):
        """Save model every `save_interval` steps."""
        update_number = _locals["update"]  # Number of updates to policy
        step_number = update_number * batch_size  # Number of steps taken on environment

        # Note: for this to ever be true save_interval must be a multiple of batch_size
        if step_number % save_interval == 0:
            if not os.path.isdir(logdir + "/checkpoints"):
                os.makedirs(logdir + "/checkpoints")
            _locals["self"].save(logdir + "/checkpoints/{}".format(step_number))

        return True  # Returning False will stop training early

    return callback


def train(
    env, num_timesteps, hardware, logdir, save, save_interval, load, seed, domain_randomization, tensorboard, p_phi=None
):
    def make_env():
        env_out = env(use_simulator=not hardware, domain_randomization=domain_randomization, frequency=250, p_phi=p_phi)
        env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])

    set_global_seeds(seed)
    policy = MlpPolicy
    model = PPO2(
        policy=policy,
        env=env,
        n_steps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        cliprange=0.2,
        verbose=1,
        tensorboard_log=tensorboard,
    )
    #store metadata of the run
    metadata = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain_randomization": domain_randomization,
                "transfer_learned_from:": load,
                "p_phi": (p_phi.mean.tolist(), p_phi.cov.tolist()) if p_phi is not None else None,
                "config": load_config()}
    os.makedirs(logdir, exist_ok=True)
    with open(logdir + "/metadata.json", "w") as f:
        json.dump(metadata, f)   
        
    # Save the model every `save_interval` steps
    if save and save_interval > 0:
        callback = init_save_callback(logdir, 2048, save_interval)
    else:
        callback = None

    # Optionally load before or save after training
    if load is not None:
        model.load_parameters(load)
    model.learn(total_timesteps=num_timesteps, callback=callback)
    if save:
        model.save(logdir + "/model")

    return model, env


def main():
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

    # Parse command line args
    parser = arg_parser()
    parser.add_argument("-e", "--env", choices=list(envs.keys()), required=True)
    parser.add_argument("-ns", "--num-timesteps", type=str, default="1e6")
    parser.add_argument("-hw", "--use-hardware", action="store_true")
    parser.add_argument("-ld", "--logdir", type=str, default="logs")
    # parser.add_argument("-v", "--video", type=str, default=None) # Doesn't work with vpython
    parser.add_argument("-l", "--load", type=str, default=None)
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-si", "--save-interval", type=float, default=5e4)
    parser.add_argument("-p", "--play", action="store_true")
    parser.add_argument("-sd", "--seed", type=int, default=-1)
    parser.add_argument("-dr", "--domain-randomization", action="store_true")
    parser.add_argument(
        "-o",
        "--output-formats",
        nargs="*",
        default=["stdout", "log", "csv", "tensorboard"],
    )
    args = parser.parse_args()

    # Set default seed
    if args.seed == -1:
        seed = np.random.randint(1, 1000)
        print("Seed is", seed)
    else:
        seed = args.seed

    device_type = "hardware" if args.use_hardware else "simulator"
    logdir = "{}/{}/{}/{}/seed-{}".format(
        args.logdir, device_type, args.env, args.num_timesteps, str(seed)
    )

    tb_logdir = logdir + "/tb"
    logger.configure(logdir, args.output_formats)

    # Round save interval to a multiple of 2048
    save_interval = int(np.ceil(args.save_interval / 2048)) if args.save else 0

    # Run training script (+ loading/saving)
    model, env = train(
        envs[args.env],
        num_timesteps=int(float(args.num_timesteps)),
        hardware=args.use_hardware,
        logdir=logdir,
        save=args.save,
        save_interval=save_interval,
        load=args.load,
        seed=seed,
        domain_randomization=args.domain_randomization,
        tensorboard=tb_logdir if "tensorboard" in args.output_formats else None,
    )

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            if not args.use_hardware:
                env.render()

    env.close()


if __name__ == "__main__":
    main()
