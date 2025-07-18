#Scientific computing
import numpy as np
import time
import os
import argparse

from scipy.stats import multivariate_normal
from cma import CMA

from gym_brt.envs import QubeSwingupEnv

from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


from load_config import load_config, params_from_config_dict
from train import train


def normalized_angle_diff_rad(a1, a2):
    """
    Calculate the shortest angle difference between two angles that are in the interval [-pi, pi].
    args:
        a1: The first angle in radians.
        a2: The second angle in radians.
    returns:
        The shortest angle difference in radians in the interval [-pi, pi].
    """
    diff = a1 - a2
    return (diff + np.pi) % (2 * np.pi) - np.pi

def real_rollout(env, model, use_hardware=True, load=None, deterministic_model=True, deterministic_resets=True, p_phi=None):
    """
    Run a rollout of the trained model in the environment.
    args:
        env: The environment to run the model in.
        model: The trained model to use for predictions.
        use_hardware: Whether to use hardware or not.
    returns:
        traj: The trajectory of the rollout.
    """
    def make_env():
        env_out = env(use_simulator=not use_hardware, frequency=250, deterministic_resets=deterministic_resets, p_phi=p_phi)
        return env_out
    try:
        env = DummyVecEnv([make_env])

        if load is not None:
            policy = MlpPolicy
            model = PPO2(policy=policy, env=env)
            model.load_parameters(load)

        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        traj = [obs.copy()]
        rewards = []
        while True:
            actions, _states = model.predict(obs, deterministic=deterministic_model)
            obs[:], reward, done, _ = env.step(actions)
            traj.append(obs.copy())
            rewards.append(reward)

            if not use_hardware:
                env.render()
            if done:
                print("done")
                obs[:] = env.reset()
                traj.append(obs.copy())
                rewards.append(reward)
                break
        episode_reward = np.sum(rewards).item()
    finally:
        env.close()
    
    return np.array(traj), episode_reward

def sim_rollout(env, model, xi, render=False, deterministic_model=True, deterministic_resets=True, sim_initial_state=np.array([0, np.pi, 0, 0], dtype=np.float64), T_max=None):
    """
    Run a rollout of the trained model in the environment.
    Model and env resets are deterministic.
    """
    #trick: pass the env i distribution p_phi as usual but use p_phi~N(xi, 0)
    xi = multivariate_normal(mean=xi, cov=np.diag(np.zeros(xi.shape[0])), allow_singular=True) #TODO: this is a hack to pass a constant sample xi and should be replaced

    def make_env():
        env_out = env(use_simulator=True, frequency=250, domain_randomization=True, p_phi=xi, deterministic_resets=deterministic_resets, sim_init_state=sim_initial_state)
        return env_out

    try:
        env = DummyVecEnv([make_env])

        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        traj = [obs.copy()]
        while True:
            actions, _states = model.predict(obs, deterministic=deterministic_model)
            obs[:], reward, done, _ = env.step(actions)
            traj.append(obs.copy())
            if render:
                env.render() #NOTE: for debuigging purpose
            if T_max is not None and len(traj) >= T_max:
                print("T_max reached")
                obs[:] = env.reset()
                traj.append(obs.copy())
                break
            if done:
                print("done")
                obs[:] = env.reset()
                traj.append(obs.copy())
                break
    finally:
        env.close()
    
    return np.array(traj)

def D(traj_xi, traj_real):
    """
    Discrepancy function to compare the real and simulated trajectories.
    args:
        traj_xi: The simulated trajectory. shape: (T_sim, 1, 4)
        traj_real: The real trajectory. shape: (T_real, 1, 4)
    returns:
        D: The discrepancy between the two trajectories.
    """
    #align the two trajectories
    #T = min([traj_xi.shape[0], traj_real.shape[0]])
    #traj_xi = traj_xi[:T, :, :] #shape: (T, 1, 4)
    #traj_real = traj_real[:T, :, :] #shape: (T, 1, 4)

    #pad the shorter trajectory with ones, to avoid optimizing for early terminiation
    T_max = max(traj_xi.shape[0], traj_real.shape[0])
    pad_xi = T_max - traj_xi.shape[0]
    pad_real = T_max - traj_real.shape[0]

    traj_xi_padded = np.pad(traj_xi, ((0, pad_xi), (0, 0), (0, 0)), mode='constant', constant_values=1) # Or some other value
    traj_real_padded = np.pad(traj_real, ((0, pad_real), (0, 0), (0, 0)), mode='constant', constant_values=1)
    assert traj_xi_padded.shape == (T_max, 1, 4), f"traj_xi_padded shape mismatch: {traj_xi_padded.shape} != {(T_max, 1, 4)}"
    assert traj_real_padded.shape == (T_max, 1, 4), f"traj_real_padded shape mismatch: {traj_real_padded.shape} != {(T_max, 1, 4)}"

    #state: theta, alpha, theta_dot, alpha_dot
    diff = np.zeros((T_max, 1, 4))
    diff[..., 0] = normalized_angle_diff_rad(traj_xi_padded[..., 0], traj_real_padded[..., 0]) #theta
    diff[..., 1] = normalized_angle_diff_rad(traj_xi_padded[..., 1], traj_real_padded[..., 1]) #alpha
    diff[..., 2:] = traj_xi_padded[..., 2:] - traj_real_padded[..., 2:] #theta_dot, alpha_dot 
    
    #Constants
    wl1 = 0.5
    wl2 = 1.0
    W = np.array([1, 1, 0.1, 0.1]) #theta, alpha, theta_dot, alpha_dot, dim: (4,)
    diff = traj_xi_padded - traj_real_padded #(T_max, 1, 4)
    #W*diff -> (4,) * (T_max, 1, 4) = (T_max, 1, 4)
    assert diff.shape == (T_max, 1, 4), f"Diff shape mismatch: {diff.shape} != {(T_max, 1, 4)}"
    assert np.linalg.norm(W*diff, ord=1, axis=2).shape == (T_max, 1), f"Weighted diff shape mismatch: {np.linalg.norm(W*diff, ord=1, axis=2).shape} != {(T_max, 1)}"
    assert np.linalg.norm(W*diff, ord=2, axis=2).shape == (T_max, 1), f"Weighted diff shape mismatch: {np.linalg.norm(W*diff, ord=2, axis=2).shape} != {(T_max, 1)}"
    
    l1_term = np.sum(np.linalg.norm(W*diff, ord=1, axis=2)) #dim: (T_max, 1) before sum
    l2_term = np.sum(np.power(np.linalg.norm(W*diff, ord=2, axis=2), 2)) #dim: (T_max, 1) before sum
    D = wl1*l1_term + wl2*l2_term

    assert diff.shape == traj_xi_padded.shape, f"Diff shape mismatch: {diff.shape} != {traj_xi_padded.shape}"
    assert (W*diff).shape == (traj_xi_padded.shape[0], traj_xi_padded.shape[1], traj_xi_padded.shape[2]), f"Weighted diff shape mismatch: {(W*diff).shape} != {(traj_xi_padded.shape[0], traj_xi_padded.shape[1], traj_xi_padded.shape[2])}"
    assert np.isscalar(l1_term), f"l1_term is not a scalar: {l1_term}"
    assert np.isscalar(l2_term), f"l2_term is not a scalar: {l2_term}"
    assert np.isscalar(D), f"D is not a scalar: {D}"
    
    return D

def create_fitness_fn(traj_real, policy, deterministic_sim_resets=True, deterministic_sim_model=True, sim_initial_state=np.array([0,np.pi,0,0], dtype=np.float64), T_max=None):
    """
    Create a fitness function compatible with TF1 CMA-ES.

    Args:
        traj_real: The real trajectory. np.ndarray.
        policy: The "policy" or openai gym model to be used for the simulation. 

    Returns:
        fitness_fn: A function that takes a symbolic tf.Tensor (M, N) and
                    returns a symbolic tf.Tensor (M,) for fitness values.
    """

    def _numpy_fitness_calculator(xi_batch):
        """
        Args:
            xi_batch: NumPy array of shape (M, N) from tf.py_func.
                                    M = population size, N = solution dimension.
        Returns:
            NumPy array of shape (M,) containing fitness values of dtype float32.
        """
        num_solutions = xi_batch.shape[0]
        D_values = np.empty(num_solutions, dtype=np.float32)

        for i in range(num_solutions):
            current_xi = xi_batch[i, :]
            traj_current_xi = sim_rollout(QubeSwingupEnv, 
                                          policy, 
                                          xi=current_xi, 
                                          deterministic_model=deterministic_sim_model, 
                                          deterministic_resets=deterministic_sim_resets, 
                                          sim_initial_state=sim_initial_state,
                                          T_max=T_max
                                        )
            
            D_values[i] = D(traj_current_xi, traj_real)
        
        return D_values # Shape (M,) np.ndarray with dtype float32 matching Tout in tf.py_func

    def fitness_fn_graph_compatible(xi_symbolic_tensor):
        """
        Args:
          xi_symbolic_tensor: tf.Tensor (symbolic, from CMA-ES graph) of shape (M, N)
        
        Returns:
          Fitness evaluations: tf.Tensor (symbolic) of shape (M,)
        """
        # tf.py_func embeds a Python function as an operation in the TensorFlow graph.
        fitness_values_op = tf.py_func(
            func=_numpy_fitness_calculator,
            inp=[xi_symbolic_tensor],  # List of input Tensors to the Python function
            Tout=tf.float32,           # TensorFlow dtype of the output(s)
            name="numpy_fitness_calculator_py_func" # Optional name
        )
        
        fitness_values_op.set_shape([None]) # Indicates a rank-1 tensor (vector) of unknown length

        return fitness_values_op

    return fitness_fn_graph_compatible

def format_array(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ", max_line_width=np.inf)

def relative_error(input_vec, target_vec):
    input_vector = np.asarray(input_vec, dtype=float)
    target_vector = np.asarray(target_vec, dtype=float)
    
    assert input_vector.shape == target_vector.shape, "Input and target vectors must have the same shape."

    abs_diff = np.abs(input_vector - target_vector)
    abs_target = np.abs(target_vector) + 1e-8  # Avoid division by zero
    relative_errors = abs_diff / abs_target
    
    return np.mean(relative_errors)

def log_progress_callback(cma_instance, ignored_standard_logger):
    generation = cma_instance.generation

    if generation == 0:
        logger.log("CMA-ES Initialization Complete. Starting search...")
        cma_instance.t0 = time.time()
    else:
        try:
            current_best_fitness = cma_instance.best_fitness() # Gets fitness of current mean m
            current_mean = cma_instance.get_mean()
            cov_matrix = cma_instance.get_covariance_matrix()
            population_size = cma_instance.population_size_py if cma_instance.population_size_py else np.floor(8 + 3*np.log(cma_instance.dimension))
            cma_instance.t1 = time.time()
            elapsed_time = cma_instance.t1 - cma_instance.t0
            cma_instance.t0 = cma_instance.t1
            #logger.log(f"Generation: {generation:4d} \nPopulation size: {population_size} \nFitness: {current_best_fitness:.6e} \nCov diag mean: {np.mean(np.diag(cov_matrix))} \nTime: {elapsed_time}s")
            logger.log(f"Generation: {generation:4d} | Population size: {population_size} | Time: {elapsed_time:.2f}s")
            logger.log(f"Current mean: {format_array(current_mean)}")
            if cma_instance.goal_solution is not None:
                logger.log(f"Goal solution: {format_array(cma_instance.goal_solution)}")
                logger.log(f"Relative error: {relative_error(current_mean, cma_instance.goal_solution):.6e}\n")
            logger.log(f"Fitness: {current_best_fitness:.6e}")


        except Exception as e:
            logger.error(f"Error fetching info in callback for generation {generation}: {e}")
            print(f"Generation: {generation:4d} | Info: [Error fetching]")


def main():

    #args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N-simopt",
        type=int,
        default=5,
        help="Number of SimOpt iterations",
    )
    parser.add_argument(
        "--T-max",
        type=int,
        default=2048,
        help="Max number of steps per episode during CMA-ES search",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=50,
        help="Number of generations for CMA-ES search",
    )
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
        help="Number of samples to use for computing the reward after each SimOpt iteration",
    )
    parser.add_argument(
        "--T-start",
        type=int,
        default=0,
        help="discard the first T_start samples of trajectories",
    )
    parser.add_argument(
        "--use-hardware",
        "-hw",
        action="store_true",
        help="Use hardware or not for the 'real' system",
    )
    args = parser.parse_args()

    os.environ["QUANSER_HW"] = "qube_servo3_usb"
    config = load_config("config.yaml")
    physical_params = params_from_config_dict(config)
    train_on_params = physical_params.copy()
    Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g = physical_params
    if not args.use_hardware:
        real_rollout_params = physical_params.copy() + np.array([0,0,0,1.0*mr,1.0*Lr,0,1.0*mp,1.0*Lp,0,0], dtype=np.float64) #Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g
        p_real_rollout = multivariate_normal(mean=real_rollout_params, cov=np.diag(real_rollout_params*0), seed=42, allow_singular=True)
    # ------------- SimOpt Initialization ---------------

    sigma_squared = np.diag((train_on_params*0.1)**2) #Set the std deviation to 20% of the mean
    phi = (train_on_params, sigma_squared)
    p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)
    
    # Loop to find an unused seed
    env_name = "QubeSwingupEnv"
    TEST_ID = 1111
    experiment_name = "simopt_sim2real_all_params" + "_" + str(TEST_ID)
    version = "v1"
    while True:
        seed = np.random.randint(1, 1000)
        base_logdir = f"logs/SimOpt/{env_name}/{experiment_name}/{version}/seed-{seed}"
        if not os.path.exists(base_logdir):
            set_global_seeds(seed)
            break

    for i_simopt in range(args.N_simopt):
        t0_simopt = time.time()    
        logdir = f"{base_logdir}/iter-{i_simopt}"
        if i_simopt >= 1:
            load = f"{base_logdir}/iter-{i_simopt-1}/model.pkl"
        else:
            load = None
        logger.configure(logdir, ["stdout", "log", "csv", "tensorboard"])

        #line4: env <- Simulatioin(p_phi)
        #line5: pi_theta_p_phi <- RL(env)
        logger.log(f"Training RL agent on env with p_phi~N({p_phi.mean}, {p_phi.cov})")
        #if i_simopt == 0:
        #    load = '/home/jonas/Masteroppgave/qube-baselines/logs/simulator/QubeSwingupEnv/3e6/seed-857/model.pkl'
        #    logger.log(f"Loading model from {load}")
        model, env = train(
            env=QubeSwingupEnv,
            num_timesteps=2000000 if i_simopt == 0 else 1000000,
            hardware=False,
            logdir=logdir,
            save=True,
            save_interval=int(np.ceil(args.save_interval / 2048)),
            load=load,
            seed=seed,
            domain_randomization=True,
            tensorboard=None,
            p_phi=p_phi
        )
        env.close()
        logger.log("Training complete. Starting rollouts...")
        #line6: tau_real <- RealRollout(pi_theta_p_phi)
        logger.log(f"Real rollout parameters: {format_array(real_rollout_params) if not args.use_hardware else 'N/A'}")
        traj_real, episode_reward = real_rollout(QubeSwingupEnv, model, use_hardware=args.use_hardware, deterministic_resets=True, deterministic_model=True, p_phi=p_real_rollout if not args.use_hardware else None)
        logger.log(f"Real rollout complete. | SimOpt Iteration: {i_simopt} | Real rollout episode reward: {episode_reward} | Time: {time.time() - t0_simopt:.2f}s")
        #line7: xi <- p_phi.sample()
        #line8: tau_xi <- SimRollout(pi_theta_p_phi, xi)
        traj_real = traj_real[:args.T_max, :, :]
        sim_init_state = traj_real[0, 0, :].copy()
        fitness_fn = create_fitness_fn(traj_real, model, deterministic_sim_resets=True, deterministic_sim_model=True, T_max=args.T_max, sim_initial_state=sim_init_state) #sim_initial_state=traj_real[0, 0, :], 

        cma_t0 = time.time()
        #previous cma instance
        #cma_cov = cma.get_covariance_matrix() if 'cma' in locals() else phi[1]
        cma_cov=phi[1]
        lower = np.maximum(0, phi[0] - 3*np.sqrt(np.diag(cma_cov)))
        upper = np.maximum(0, phi[0] + 3*np.sqrt(np.diag(cma_cov)))
        bounds = [[lower[i], upper[i]] if i in [3,4,6,7,9] else [phi[0][i], phi[0][i]] for i in range(len(lower))] #only allow the parameters mr, Lr, mp, Lp to vary, the rest are fixed
        cma = CMA(
            initial_solution=p_phi.mean.tolist(),
            initial_step_size=np.mean(np.sqrt(np.diag(phi[1]))),
            fitness_function=fitness_fn,
            enforce_bounds=bounds,
            termination_no_effect=1e-8,
            callback_function=log_progress_callback,
        )
        cma.goal_solution = real_rollout_params if not args.use_hardware else None
        best_solution, best_fitness = cma.search(max_generations=args.max_generations)
        logger.log(f"CMA-ES search complete. | SimOpt Iteration: {i_simopt} | Best solution: {best_solution} | Best fitness: {best_fitness} | Time: {time.time() - cma_t0:.2f}s")
        logger.log(f"CMA solution: {format_array(best_solution)}")
        logger.log(f"Known solution: {format_array(real_rollout_params) if not args.use_hardware else 'N/A'}")
        #update p_phi
        phi = (cma.get_mean(), phi[1])
        p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)
        logger.log(f"Updated p_phi~N({format_array(p_phi.mean)}, {format_array(np.diag(p_phi.cov))})")

        #Meassure avg reward on the real rollout
        deterministic_resets = False
        deterministic_model = False
        logger.log(f"Rolling out to REAL {'HARDWARE' if args.use_hardware else 'SIMULATOR'} with deterministics resets: {deterministic_resets} and deterministic model: {deterministic_model}")
        rewards = []
        for i in range(args.reward_samples): 
            _, reward = real_rollout(QubeSwingupEnv, model, use_hardware=args.use_hardware, deterministic_resets=deterministic_resets, deterministic_model=deterministic_model, p_phi=p_real_rollout if not args.use_hardware else None)
            rewards.append(reward)
        with open(f"{logdir}/reward.txt", "w") as f:
            f.write(f"real_rollouts_rewards: {rewards}\n")
            f.write(f"mean_reward: {np.mean(rewards)}\n")
            f.write(f"std_reward: {np.std(rewards)}\n")
        f.close()
        
        logger.log(f"Finished SimOpt Iteration: {i_simopt} | Time: {time.time() - t0_simopt:.2f}s")
    

    """
    # ------------- SimOpt Main Loop ----------------
    for i in range(N_simopt):
        env = QubeSwingupEnv(dist=p_phi, frequency=...)
        policy = train(env)

        traj_real = RealRollout(policy)
        fitness_func = create_fitness_fn(traj_real, policy)
        best_solution, best_fitness = cma.search()
        phi = (best_solution, SIGMA) #find a way to extract an appropriate sigma

    """
    
if __name__ == "__main__":
    main()
