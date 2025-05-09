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
        env_out = env(use_simulator=not use_hardware, frequency=250)
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
            actions = model.step(obs)[0]
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

def sim_rollout(env, model, xsi):
    #trick: pass the env i distribution p_phi as usual but use p_phi~N(xi, 0)
    xsi = multivariate_normal(mean=xsi, cov=np.diag(np.zeros(xsi.shape[0])), allow_singular=True) #TODO: this is a hack to pass a constant sample xsi and should be replaced

    def make_env():
        env_out = env(use_simulator=True, frequency=250, domain_randomization=True, p_phi=xsi)
        return env_out

    try:
        env = DummyVecEnv([make_env])

        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        traj = [obs.copy()]
        while True:
            actions = model.step(obs)[0]
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

    #Constants
    wl1 = 0.5
    wl2 = 1.0
    W = np.array([1, 1, 0.5, 0.5]) #theta, alpha, theta_dot, alpha_dot, dim: (4,)
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

    N_simopt = 4 #number of SimOpt iterations
    sigma = np.diag(np.ones(mu.shape[0])*0.000025) #0.5 as initial value is taken from paper.
    phi = (mu, sigma)
    p_phi = multivariate_normal(mean=phi[0], cov=phi[1], seed=42)

    # Loop to find an unused seed
    env_name = "QubeSwingupEnv"
    while True:
        seed = 666#np.random.randint(1, 1000)
        base_logdir = f"logs/SimOpt/{env_name}/seed-{seed}"
        if not os.path.exists(base_logdir):
            set_global_seeds(seed)
            break
    save_interval = 5e4

    for i in range(N_simopt):    
        logdir = f"{base_logdir}/iter-{i}"
        if i >= 1:
            load = f"{base_logdir}/iter-{i-1}/model.pkl"
        else:
            load = None
        logger.configure(logdir, ["stdout", "log", "csv", "tensorboard"])
        
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
        #line7: xsi <- p_phi.sample()
        xsi = np.array([p_phi.rvs(size=1)])
        print("xsi: ", xsi)
        #line8: tau_xsi <- SimRollout(pi_theta_p_phi, xsi)
        traj_xsi = sim_rollout(QubeSwingupEnv, model, xsi=xsi)


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
