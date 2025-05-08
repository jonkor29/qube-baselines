#Scientific computing
import numpy as np
import time

from scipy.stats import multivariate_normal

from gym_brt.envs import QubeSwingupEnv

from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from load_config import load_config

config = load_config("config.yaml")
mu = np.array([config['mp']]) #mean of the distribution
print("mu: ", mu)
# ------------- SimOpt Initialization ---------------

N_simopt = 10 #number of SimOpt iterations
sigma = np.diag(np.ones(mu.shape[0])*0.00001) #0.5 as initial value is taken from paper.
phi = (mu, sigma)
p_phi = multivariate_normal(mean=phi[0], cov=phi[1])
sample = p_phi.rvs(size=1)
print("sample: ", sample)

env = QubeSwingupEnv(domain_randomization=True, use_simulator=True, p_phi=p_phi)

for i in range(10):
    input("Press Enter to continue...")
    env.reset()
    env.step(env.action_space.sample())
    print("action: ", env.action_space.sample())
    print("params:", env.get_physical_params())


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