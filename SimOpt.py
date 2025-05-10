#Scientific computing
import numpy as np
import time
import os

from scipy.stats import multivariate_normal

from gym_brt.envs import QubeSwingupEnv

from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2 import PPO2



from load_config import load_config
from train import train

def normalized_angle_diff_rad(a1, a2):
    diff = a1 - a2
    # Ensure result is in [-pi, pi]
    return (diff + np.pi) % (2 * np.pi) - np.pi

def real_rollout(env, model, use_hardware=True, load=None):
    """
    Run a rollout of the trained model in the environment.
    args:
        env: The environment to run the model in.
        model: The trained model to use for predictions.
        use_hardware: Whether to use hardware or not.
    returns:
        traj: The trajectory of the rollout.
    """
    # Parse command line args
    def make_env():
        env_out = env(use_simulator=not use_hardware, frequency=250, deterministic_resets=True)
        return env_out
    try:
        env = DummyVecEnv([make_env])

        if load is not None:
            policy = MlpPolicy
            model = PPO2(policy=policy, env=env)
            model.load_parameters(load)

        print("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        traj = [obs.copy()]
        while True:
            actions, _states = model.predict(obs, deterministic=True)
            obs[:], reward, done, _ = env.step(actions)
            traj.append(obs.copy())

            if not use_hardware:
                env.render()
            if done:
                print("done")
                obs[:] = env.reset()
                traj.append(obs.copy())
                break
    finally:
        env.close()
    
    return np.array(traj)

def sim_rollout(env, model, xi):
    #trick: pass the env i distribution p_phi as usual but use p_phi~N(xi, 0)
    xi = multivariate_normal(mean=xi, cov=np.diag(np.zeros(xi.shape[0])), allow_singular=True) #TODO: this is a hack to pass a constant sample xi and should be replaced

    def make_env():
        env_out = env(use_simulator=True, frequency=250, domain_randomization=True, p_phi=xi, deterministic_resets=True)
        return env_out

    try:
        env = DummyVecEnv([make_env])

        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        traj = [obs.copy()]
        while True:
            actions, _states = model.predict(obs, deterministic=True)
            obs[:], reward, done, _ = env.step(actions)
            traj.append(obs.copy())
            env.render() #NOTE: for debuigging purpose

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
    T = min([traj_xi.shape[0], traj_real.shape[0]])
    traj_xi = traj_xi[:T, :, :] #shape: (T, 1, 4)
    traj_real = traj_real[:T, :, :] #shape: (T, 1, 4)

    #calculate the difference between the two trajectories
    #state: theta, alpha, theta_dot, alpha_dot
    diff = np.zeros((T, 1, 4))
    diff[..., 0] = normalized_angle_diff_rad(traj_xi[..., 0], traj_real[..., 0]) #theta
    diff[..., 1] = normalized_angle_diff_rad(traj_xi[..., 1], traj_real[..., 1]) #alpha
    diff[..., 2:] = traj_xi[..., 2:] - traj_real[..., 2:] #theta_dot, alpha_dot 
    #Constants
    wl1 = 0.5
    wl2 = 1.0
    W = np.array([1, 1, 0, 0]) #theta, alpha, theta_dot, alpha_dot, dim: (4,)
    diff = traj_xi - traj_real #(T, 1, 4)
    #W*diff -> (4,) * (T, 1, 4) = (T, 1, 4)
    assert diff.shape == (T, 1, 4), f"Diff shape mismatch: {diff.shape} != {(T, 1, 4)}"
    assert np.linalg.norm(W*diff, ord=1, axis=2).shape == (T, 1), f"Weighted diff shape mismatch: {np.linalg.norm(W*diff, ord=1, axis=2).shape} != {(T, 1)}"
    assert np.linalg.norm(W*diff, ord=2, axis=2).shape == (T, 1), f"Weighted diff shape mismatch: {np.linalg.norm(W*diff, ord=2, axis=2).shape} != {(T, 1)}"
    
    l1_term = np.sum(np.linalg.norm(W*diff, ord=1, axis=2)) #dim: (T, 1) before sum
    l2_term = np.sum(np.power(np.linalg.norm(W*diff, ord=2, axis=2), 2)) #dim: (T, 1) before sum
    D = wl1*l1_term + wl2*l2_term

    assert diff.shape == traj_xi.shape, f"Diff shape mismatch: {diff.shape} != {traj_xi.shape}"
    assert (W*diff).shape == (traj_xi.shape[0], traj_xi.shape[1], traj_xi.shape[2]), f"Weighted diff shape mismatch: {(W*diff).shape} != {(traj_xi.shape[0], traj_xi.shape[1], traj_xi.shape[2])}"
    assert np.isscalar(l1_term), f"l1_term is not a scalar: {l1_term}"
    assert np.isscalar(l2_term), f"l2_term is not a scalar: {l2_term}"
    assert np.isscalar(D), f"D is not a scalar: {D}"
    
    return D

def main():
    config = load_config("config.yaml")
    mu = np.array([config['mp']]) #mean of the distribution
    print("mu: ", mu)
    # ------------- SimOpt Initialization ---------------

    N_simopt = 10 #number of SimOpt iterations
    sigma = np.diag(np.ones(mu.shape[0])*0.000025) #0.5 as initial value is taken from paper.
    phi = (mu, sigma)
    p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42, allow_singular=True)

    # Loop to find an unused seed
    env_name = "QubeSwingupEnv"
    while True:
        seed = 666#np.random.randint(1, 1000)
        base_logdir = f"logs/SimOpt/{env_name}/seed-{seed}"
        if not os.path.exists(base_logdir):
            set_global_seeds(seed)
            break
    save_interval = 5e4

    Ds = []
    avg_diffs = []
    sum_diffs = []
    for i in range(N_simopt):    
        logdir = f"{base_logdir}/iter-{i}"
        if i >= 1:
            load = f"{base_logdir}/iter-{i-1}/model.pkl"
        else:
            load = None
        logger.configure(logdir, ["stdout", "log", "csv", "tensorboard"])
        load = '/home/jonas/Masteroppgave/qube-baselines/logs/simulator/QubeSwingupEnv/3e6/seed-667/model.pkl'#TODO: remove
        #line4: env <- Simulatioin(p_phi)
        #line5: pi_theta_p_phi <- RL(env)
        model, env = train(
            env=QubeSwingupEnv,
            num_timesteps=2048,
            hardware=False,
            logdir=logdir,
            save=True,
            save_interval=int(np.ceil(save_interval / 2048)),
            load=load,
            seed=seed,
            domain_randomization=True,
            tensorboard=None,
            p_phi=p_phi
        )
        env.close()

        #line6: tau_real <- RealRollout(pi_theta_p_phi)
        traj_real = real_rollout(QubeSwingupEnv, model, use_hardware=False)
        #line7: xi <- p_phi.sample()
        xi = np.array([0.024])#np.array([p_phi.rvs(size=1)])
        print("xi: ", xi)
        #line8: tau_xi <- SimRollout(pi_theta_p_phi, xi)
        traj_xi = sim_rollout(QubeSwingupEnv, model, xi=xi)

        print("traj_xi: ", traj_xi)
        print("traj_real: ", traj_real)
        #calculate average differences for the trajectories along the time axis, for each dimension
        T = min([traj_xi.shape[0], traj_real.shape[0]])
        traj_xi_length = traj_xi.shape[0]
        traj_real_length = traj_real.shape[0]
        print("traj_xi_length: ", traj_xi_length)
        print("traj_real_length: ", traj_real_length)
        traj_xi = traj_xi[:T, :, :]
        traj_real = traj_real[:T, :, :]
        avg_diffs.append(np.mean(np.abs(traj_xi - traj_real), axis=0))

        #calculate the diff using the angle difference
        sum_diff = np.zeros((traj_xi.shape[0], 1, 4))
        sum_diff[..., 0] = normalized_angle_diff_rad(traj_xi[..., 0], traj_real[..., 0])
        sum_diff[..., 1] = normalized_angle_diff_rad(traj_xi[..., 1], traj_real[..., 1])
        sum_diff[..., 2:] = traj_xi[..., 2:] - traj_real[..., 2:]
        sum_diffs = np.sum(np.abs(traj_xi - traj_real), axis=0)
        Ds.append(D(traj_xi, traj_real))
        print("D's: ", Ds)
        print("avg_diffs: ", avg_diffs)
        print("sum_diffs: ", sum_diffs)
        print("avg D: ", np.mean(Ds))
    

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
