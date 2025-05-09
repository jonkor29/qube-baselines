#Scientific computing
import numpy as np
import time

from scipy.stats import multivariate_normal

from gym_brt.envs import QubeSwingupEnv

from stable_baselines import logger

from load_config import load_config
from train import train

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