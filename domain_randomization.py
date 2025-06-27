#Scientific computing
import numpy as np
import os
import argparse

from scipy.stats import multivariate_normal

from gym_brt.envs import QubeSwingupEnv

from stable_baselines import logger
from stable_baselines.common import set_global_seeds

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


from load_config import load_config, params_from_config_dict
from train import train

def format_array(arr):
    return np.array2string(arr, precision=10, suppress_small=True, separator=", ", max_line_width=np.inf)

def main():
    """
    This script trains an agent in QubeSwingupEnv using domain randomization.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5e4,
        help="Number of steps between saving the model",
    )
    parser.add_argument(
        "--reward-samples",
        type=int,
        default=10,
        help="Number of samples to use for computing the reward after training",
    )
    parser.add_argument(
        "--params-to-randomize",
        nargs='+',
        default=['Rm', 'kt', 'km', 'mr', 'Lr', 'Dr', 'mp', 'Lp', 'Dp', 'g'],
        help="A list of the physical parameter names to randomize (e.g., mp Lp Dr). If empty, all parameters are randomized."
    )
    parser.add_argument(
        "--standard-deviation-percentage",
        "-std",
        type=float,
        default=0.1,
        help="Percentage of the mean value to use as the standard deviation for the randomization of the parameters. Default is 10%."
    )
    args = parser.parse_args()

    # Loop to find an unused seed to open a new log directory
    env_name = "QubeSwingupEnv"
    vars_randomized = "all"
    std_dev_percentage = args.standard_deviation_percentage
    while True:
        seed = np.random.randint(1, 1000)
        base_logdir = f"logs/domain_randomization/{env_name}/{vars_randomized}/{std_dev_percentage}/seed-{seed}"
        if not os.path.exists(base_logdir):
            set_global_seeds(seed)
            break

    #set up logging
    logdir = base_logdir
    logger.configure(logdir, ["stdout", "log", "csv", "tensorboard"])

    os.environ["QUANSER_HW"] = "qube_servo3_usb"
    config = load_config("config.yaml")
    physical_params = params_from_config_dict(config)
    param_names = ['Rm', 'kt', 'km', 'mr', 'Lr', 'Dr', 'mp', 'Lp', 'Dp', 'g']
    param_map = {name: i for i, name in enumerate(param_names)}

    # ------------- DOMAIN RANDOMIZATION Initialization ---------------
    stds = np.zeros_like(physical_params)
    for param_name in args.params_to_randomize:
        if param_name in param_map:
            param_index = param_map[param_name]
            # Set std dev to 10% of the mean value ONLY for the selected parameter
            stds[param_index] = args.standard_deviation_percentage * physical_params[param_index] 
        else:
            logger.log(f"Warning: Parameter '{param_name}' not recognized. Skipping.")
    logger.log(f"Parameters to randomize: {args.params_to_randomize}")
    logger.log(f"stds={format_array(stds)}")
    
    sigma_squared = np.diag(stds**2) # Variances are the square of standard deviations
    phi = (physical_params, sigma_squared)
    p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)
    logger.log(f"Domain Randomization parameters: mean={format_array(p_phi.mean)}, cov={format_array(stds**2)}")
    # ------------- DOMAIN RANDOMIZATION End ---------------
    model, env = train(
        env=QubeSwingupEnv,
        num_timesteps=2000000,
        hardware=False,
        logdir=logdir,
        save=True,
        save_interval=int(np.ceil(args.save_interval / 2048)),
        load=None,
        seed=seed,
        domain_randomization=True,
        tensorboard=None,
        p_phi=p_phi
    )
    env.close()
    logger.log(f"Finished Domain Randomized Training")


    
if __name__ == "__main__":
    main()

"""
Example usage:

python domain_randomization.py --params-to-randomize mp --standard-deviation-percentage 0.5
python domain_randomization.py --params-to-randomize mp Lp mr Lr --standard-deviation-percentage 0.1
python domain_randomization.py -std 0.1
"""