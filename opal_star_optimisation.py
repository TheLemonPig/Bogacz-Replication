import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import simulations1 as sim
import pickle
import math
import scipy.optimize as opt
import matplotlib.pylab as pl
from datetime import datetime

# distribution = input('Distribution: ') # Bernoulli or Gaussian; use Gaussian for the Gershman task
# task = input('Task: ') # bernoulli1, bernoulli2, bernoulli3, gaussian, gershman
# random_mean = input('Random mean: ') # True or False; True for the Gershman task only

# random_mean = bool(random_mean)
distribution = 'Bernoulli'  # 'Gaussian'
task = 'bernoulli3'  # 'gaussian'
random_mean = False

params_bernoulli1 = np.array([.45, .45, .45, .45, .45, .45, .45, .45, .45, .55])
params_bernoulli3 = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .9])
params_bernoulli2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, .3])

params_gaussian = np.array([[.45, .45, .45, .45, .45, .45, .45, .45, .45, .55],
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])

env_params = params_bernoulli3  # params_gaussian
print(env_params)

trials = 800
reps = 800

mean_regret = lambda params: np.mean(sim.run(distribution,
                                             env_params,
                                             'opal-star',
                                             # [0.1, 0.5, params],
                                             [params[0], params[1], params[2]],
                                             reps,
                                             trials)['regrets'])
sampled_points = []
options = {
    'ftol': 1e-8,    # Function tolerance for termination
    'maxev': 500,    # Maximum number of function evaluations
    'maxtime': 300,   # Maximum time in seconds
    'minimize_every_iter': True,  # Minimize after every iteration
    'disp': True
}
# Define a callback function to record the sampled points
def callback(x):
    sampled_points.append(x)

min_regret = np.inf
best_params = []
method = 'L-BFGS-B'
constraint = None
# param_ranges = ((0, 10))
param_ranges = ((0.01, 0.3), (0.01, 0.3), (0.01, 0.3))
# param_ranges = ((0, 1), (0, 1), (0, 10))
# n = 100 # 2**len(param_ranges) + 1
start = datetime.now()
optimal_params = opt.shgo(mean_regret, param_ranges, options=options, minimizer_kwargs={'method': method}, constraints=constraint, iters=10,
                          callback=callback)

# initial_guess = np.array([p_range[1] for p_range in param_ranges])
# result = opt.minimize(mean_regret, initial_guess, bounds=param_ranges)
# optimal_params = result.x
end = datetime.now()
print(start)
print(end)
print(f'Optimization time: {np.round((end-start).total_seconds()/3600, decimals=2)} hours')
print(f'Trials: {trials} , Reps: {reps}')

# Print the optimization result
print("\nOptimization Result:")
print(optimal_params)

# print("\nLocal Minima:")
# print(f'{optimal_params.xl.shape[0]} {optimal_params.funl.shape[0]}')
# for i in range(min(optimal_params.xl.shape[0], optimal_params.funl.shape[0])):
#     print(f'{optimal_params.xl[i]}: {np.round(optimal_params.funl[i], decimals=5)}')

# Print the trace of sampled points
# print("\nTrace of Sampled Points:")
# for point in sampled_points:
#     print(point)

# for alpha_c in (0.025, 0.05, 0.1):
#
#     for alpha_gn in np.arange(0.1, 1.01, 0.05):
#
#         for beta in np.arange(1, 10.1, 0.5):
#
#             regret = mean_regret([alpha_c, alpha_gn, beta])
#
#             if regret < min_regret:
#                 min_regret = regret
#                 best_params = [alpha_c, alpha_gn, beta]
#
#                 print('NEW MINIMUM:')
#                 print(min_regret)
#                 print(best_params)
