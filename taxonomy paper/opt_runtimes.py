from pynumdiff.utils.simulate import pi_cruise_control, sine, triangle
from pynumdiff.kalman_smooth import convex_smooth, rtsdiff
from pynumdiff.utils import evaluate
from scipy.linalg import expm
from time import time
from matplotlib import pyplot
import numpy as np
import pandas

order = 2
q = 1e6
r = 0.1
meas_huberM = 1
proc_huberM = 2
dt = 0.01
durations = [4] #np.arange(1, 20)*4
sim = pi_cruise_control

sparse_runtimes = []
clarabel_runtimes = []
rts_runtimes = []
for duration in durations:
	x, x_truth, dxdt_truth = sim(duration=duration, noise_type='normal', noise_parameters=[0, 0.1], dt=dt, outliers=True)

	A_c = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
	Q_c = np.zeros(A_c.shape); Q_c[-1,-1] = q # continuous-time uncertainty around the last derivative
	C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
	R = np.array([[r]]) # 1 observed state, so this is 1x1

	# convert to discrete time using matrix exponential
	eM = expm(np.block([[A_c, Q_c], [np.zeros(A_c.shape), -A_c.T]]) * dt) # Note this could handle variable dt, similar to rtsdiff
	A_d = eM[:order+1, :order+1]
	Q_d = eM[:order+1, order+1:] @ A_d.T
	#if np.linalg.cond(Q_d) > 1e12: Q_d += np.eye(order + 1)*1e-12 # for numerical stability with convex solver. Doesn't change answers appreciably (or at all).

	before = time()
	#x_states1 = sparse_smooth(x, A_d, Q_d, C, R, meas_huberM)
	#x_states1 = chambolle_pock(x, A_d, Q_d, C, R, meas_huberM)
	between = time()
	x_states2 = convex_smooth(x, A_d, Q_d, C, R, proc_huberM=proc_huberM, meas_huberM=meas_huberM)
	after = time()
	rts_xhat, rts_dxhat = rtsdiff(x, dt, order, q, False)
	rtstime = time()
	print(f"cp-v2 took {between-before}s; cvxpy took {after-between}s; and RTS took {rtstime-after}s")
	sparse_runtimes.append(between-before)
	clarabel_runtimes.append(after-between)
	rts_runtimes.append(rtstime-after)

	#print(x.shape, x_states1.shape, x_states2.shape, x_truth.shape, dxdt_truth.shape)
	#evaluate.plot(x, dt, x_states1[:,0], x_states1[:,1], x_truth, dxdt_truth)
	evaluate.plot(x, dt, x_states2[0], x_states2[1], x_truth, dxdt_truth)
	#evaluate.plot(x, dt, rts_xhat, rts_dxhat, x_truth, dxdt_truth)
	pyplot.show()

# pyplot.figure()
# pyplot.plot(durations, sparse_runtimes, label='CP-V2')
# pyplot.plot(durations, clarabel_runtimes, label='cvxpy')
# pyplot.plot(durations, rts_runtimes, label='RTS')
# pyplot.legend()
# pyplot.show()

