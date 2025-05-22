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


from load_config import load_config
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

def real_rollout(env, model, use_hardware=True, load=None, deterministic_model=True, deterministic_resets=True):
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
        env_out = env(use_simulator=not use_hardware, frequency=250, deterministic_resets=deterministic_resets)
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
            logger.log(f"Generation: {generation:4d} | Population size: {population_size} Fitness: {current_best_fitness:.6e} | Mean: {current_mean[0]:.4f} | Cov: {cov_matrix} | Time: {elapsed_time}s")

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
    #os.environ["QUANSER_HW"] = "qube_servo3_usb_wrong_pendulum_mass" 
    os.environ["QUANSER_HW"] = "qube_servo3_usb"
    config = load_config("config.yaml")
    mu = np.array([config['mp']]) #mean of the distribution
    # ------------- SimOpt Initialization ---------------

    sigma_squared = np.diag(np.ones(mu.shape[0])*0.000025)
    phi = (mu, sigma_squared)
    p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)

    # Loop to find an unused seed
    env_name = "QubeSwingupEnv"
    TEST_ID = 6989
    experiment_name = "sim2sim_double_mp" + "_" + str(TEST_ID) 
    while True:
        seed = np.random.randint(1, 1000)
        base_logdir = f"logs/SimOpt/{env_name}/{experiment_name}/seed-{seed}"
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

        
        #os.environ["QUANSER_HW"] = "qube_servo3_usb_wrong_pendulum_mass"
        os.environ["QUANSER_HW"] = "qube_servo3_usb"  
        model, env = train(
            env=QubeSwingupEnv,
            num_timesteps=1000000 if i_simopt == 0 else 1000000,
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
        #force double mass when using simulator
        os.environ["QUANSER_HW"] = "qube_servo3_usb_wrong_pendulum_mass"
        #os.environ["QUANSER_HW"] = "qube_servo3_usb" 
        traj_real, episode_reward = real_rollout(QubeSwingupEnv, model, use_hardware=args.use_hardware, deterministic_resets=False, deterministic_model=True)
        logger.log(f"Real rollout complete. | SimOpt Iteration: {i_simopt} | Real rollout episode reward: {episode_reward} | Time: {time.time() - t0_simopt:.2f}s")
        #line7: xi <- p_phi.sample()
        #xi_0 = np.array([0.024])#np.array([p_phi.rvs(size=1)])
        #line8: tau_xi <- SimRollout(pi_theta_p_phi, xi)
        os.environ["QUANSER_HW"] = "qube_servo3_usb"
        #os.environ["QUANSER_HW"] = "qube_servo3_usb_wrong_pendulum_mass" 
        #traj_xi_0 = sim_rollout(QubeSwingupEnv, model, xi=xi_0)
        #NOTE: cut trajectory to first second
        traj_real = traj_real[:args.T_max, :, :]
        fitness_fn = create_fitness_fn(traj_real, model, deterministic_sim_resets=True, deterministic_sim_model=True, sim_initial_state=traj_real[0, 0, :], T_max=args.T_max)
        #fitness = fitness_fn(xi_0)
        cma_t0 = time.time()
        lower = max(0, phi[0].item() - 3*np.sqrt(phi[1]).item())
        upper = max(0, phi[0].item() + 3*np.sqrt(phi[1]).item())
        cma = CMA(
            initial_solution=p_phi.mean.tolist(),
            initial_step_size=phi[1].item(),
            fitness_function=fitness_fn,
            enforce_bounds=[[lower, upper]],
            termination_no_effect=1e-8,
            callback_function=log_progress_callback,
        )
        best_solution, best_fitness = cma.search(max_generations=args.max_generations)
        logger.log(f"CMA-ES search complete. | SimOpt Iteration: {i_simopt} | Best solution: {best_solution} | Best fitness: {best_fitness} | Time: {time.time() - cma_t0:.2f}s")
        logger.log(f"Finished SimOpt Iteration: {i_simopt} | Time: {time.time() - t0_simopt:.2f}s")
        #update p_phi
        phi = (cma.get_mean(), cma.get_covariance_matrix())
        p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)
        logger.log(f"Updated p_phi~N({p_phi.mean}, {p_phi.cov})")

        #Meassure avg reward on the real rollout
        os.environ["QUANSER_HW"] = "qube_servo3_usb_wrong_pendulum_mass"
        #os.environ["QUANSER_HW"] = "qube_servo3_usb" 
        deterministic_resets = False
        deterministic_model = True
        logger.log(f"Rolling out to REAL {'HARDWARE' if args.use_hardware else 'SIMULATOR'} with deterministics resets: {deterministic_resets} and deterministic model: {deterministic_model}")
        rewards = []
        for i in range(args.reward_samples): 
            _, reward = real_rollout(QubeSwingupEnv, model, use_hardware=args.use_hardware, deterministic_resets=deterministic_resets, deterministic_model=deterministic_model)
            rewards.append(reward)
        os.environ["QUANSER_HW"] = "qube_servo3_usb" 
        with open(f"{logdir}/reward.txt", "w") as f:
            f.write(f"real_rollouts_rewards: {rewards}\n")
            f.write(f"mean_reward: {np.mean(rewards)}\n")
            f.write(f"std_reward: {np.std(rewards)}\n")
        f.close()


        #TODO: tweak: max_gen, N_simopt, sigma, n_timesteps, for loop real rollout range()
        """        
        with open(f"{base_logdir}/best_solutions.txt", "w") as f:
        f.write(f"---------SimOpt iteration: {i_simopt}-----------\n")
        f.write(f"Best solution: {best_solution}\n")
        f.write(f"Best fitness: {best_fitness}\n")
        #f.write(f"Unoptimized fitness: {fitness}\n")
        f.write(f"Unoptimized D-value: {D(traj_xi_0, traj_real)}\n")
        f.write(f"xi_0: {xi_0}\n")
        f.write(f"traj_xi_0: {traj_xi_0}\n")
        f.write(f"traj_real: {traj_real}\n")
        f.write("-------------------------------\n")
        """

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
