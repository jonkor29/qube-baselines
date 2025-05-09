#Scientific computing
import numpy as np
import time

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


config = load_config("config.yaml")
mu = np.array([config['mp']]) #mean of the distribution
print("mu: ", mu)
# ------------- SimOpt Initialization ---------------

N_simopt = 10 #number of SimOpt iterations
sigma = np.diag(np.ones(mu.shape[0])*0.000025) #0.5 as initial value is taken from paper.
phi = (mu, sigma)
p_phi = multivariate_normal(mean=phi[0], cov=phi[1])
sample = p_phi.rvs(size=1)

seed = np.random.randint(0, 1000)
save_interval = 5e4
env_name = "QubeSwingupEnv"
base_logdir = f"logs/SimOpt/{env_name}/seed-{seed}"

for i in range(N_simopt):    
    logdir = f"{base_logdir}/iter-{i}"
    if i >= 1:
        load = f"{base_logdir}/iter-{i-1}"
    else:
        load = None
    logger.configure(logdir, ["stdout", "log", "csv", "tensorboard"])
    
    #env <- Simulatioin(p_phi)
    #pi_theta_p_phi <- RL(env)
    model, env = train(
        env=QubeSwingupEnv,
        num_timesteps=3000000,
        hardware=False,
        logdir=logdir,
        save=True,
        save_interval=int(np.ceil(save_interval / 2048)),
        load=None,
        seed=seed,
        domain_randomization=True,
        tensorboard=None,
        p_phi=p_phi
    )
    env.close()

    #tau_real <- RealRollout(pi_theta_p_phi)
    traj_real = real_rollout(QubeSwingupEnv, model, use_hardware=True)
    print("traj_real: ", traj_real.shape) #(num_timesteps, 1, 4) (middle dimension is the number of vecenvs, but we only use one env at a time)
    print("max(traj_real): ", np.max(traj_real, axis=0))
    print("min(traj_real): ", np.min(traj_real, axis=0))
    print("traj_real[0:10]", traj_real[0:10])
    print("traj_real[-10:-1]", traj_real[-10:-1])

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