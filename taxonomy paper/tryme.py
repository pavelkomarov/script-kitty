from pynumdiff.optimize import optimize
from pynumdiff.utils.simulate import sine, triangle, pop_dyn, linear_autonomous, pi_cruise_control, lorenz_x
from pynumdiff.kalman_smooth import robustdiff, rtsdiff
from pynumdiff.smooth_finite_difference import butterdiff
from pynumdiff.total_variation_regularization import tvrdiff
from pynumdiff.polynomial_fit import splinediff
from pynumdiff.finite_difference import finitediff
from pynumdiff.utils import evaluate
from matplotlib import pyplot
from time import time
import numpy as np


if __name__ == '__main__':
	# time step and time series length
	dt = 0.01 # sampling time step
	duration = 4 # sec
	#problem = linear_autonomous # select one of the options imported from the simulate module
	method = tvrdiff
	for problem in [sine, triangle, pop_dyn, linear_autonomous, pi_cruise_control, lorenz_x]:
		cutoff_frequency = 3
		tvgamma = np.exp(-1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1)
		#for problem in [sine, triangle, pop_dyn, linear_autonomous, lorenz_x]:

		x, x_truth, dxdt_truth = problem(duration, noise_type='normal',
		                                noise_parameters=[0, 0.1], dt=dt, outliers=True, random_seed=1)

		start = time()
		#params, val = optimize(method, x, dt, dxdt_truth=dxdt_truth)
		params, val = optimize(method, x, dt, tvgamma=tvgamma, huberM=2, search_space_updates=({'order':{2,3}} if problem != triangle else {}))
		print(f"that took {time() - start} seconds")
		print('Optimal parameters: ', params)
		print('Optimal loss: ', val)
		x_hat, dxdt_hat = method(x, dt, **params)
		evaluate.plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)
		# pyplot.savefig(f"boo_{problem.__name__}_{method.__name__}.png")
		print("RMSE:", evaluate.rmse(dxdt_truth, dxdt_hat))
		print("R^2:", evaluate.error_correlation(dxdt_truth, dxdt_hat))
	pyplot.show()
