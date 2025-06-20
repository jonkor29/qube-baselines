import os
import numpy as np

if os.getcwd().split(os.sep)[-1] == 'notebook':
    os.chdir('..')

from cma import CMA
import time

#create a counter that counts how many times it has been called
def counter():
    count = 0
    def inner():
        nonlocal count
        count += 1
        return count
    return inner
counter = counter()
# create a function that returns the number of calls
def fitness_fn(x):
    global counter
    for i in range(x.shape[0]):
        count = counter()
    print(f"'SimRollout' called {count} times")
    return (x[:,0]-10)**2 + 20 + x[:,1]

x = np.array([[0.0, 1.0], [10, 0]])

print("take the time of this function")
start = time.time()

cma = CMA(
    initial_solution=[1.5, -0.4],
    initial_step_size=1.0,
    fitness_function=fitness_fn,
    store_trace=True,
    enforce_bounds=[[-100, 100], [-1, 1]],
    termination_no_effect=1e-8
)

best_solution, best_fitness = cma.search()
end = time.time()
print("time taken in seconds", end - start)

print('Number of generations:', cma.generation)
print(f'Best solution: [{best_solution[0]:.5f}, {best_solution[1]:.5f}]')
print(f'Best fitness: {best_fitness:.4f}')

mu_new, sigma_new = cma.get_mean(), cma.get_covariance_matrix()
print(f"mu_new: {mu_new}")
print(f"sigma_new: {sigma_new}")