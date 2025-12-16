import numpy as np
from matplotlib import pyplot

from pynumdiff.utils.simulate import pi_cruise_control
from pynumdiff.kalman_smooth import kalman_filter, rts_smooth, convex_smooth
from pynumdiff.utils import evaluate


duration = 4
simdt = 0.01
dt = 0.01
r = 0.1
q = 1e8


y, x, dxdt = pi_cruise_control(duration=duration, noise_type='normal', noise_parameters=(0, 0.1), outliers=True,
               random_seed=1, dt=0.01, simdt=simdt)
#y[50] = np.nan

# disturbance
t = np.arange(0, duration, simdt)
slope = 0.01*(np.sin(2*np.pi*t) + 0.3*np.sin(4*2*np.pi*t + 0.5) + 1.2*np.sin(1.7*2*np.pi*t + 0.5)) # this is part of u

# parameters
mg = 10000 # mass*gravity
fr = 0.9 # friction
ki = 0.05 # integral control
kp = 0.25 # proportional control
vd = 0.5 # desired velocity

# Here state is [pos, vel, accel, cumulative pos error]
A = np.array([[1,  simdt,     (simdt**2)/2,         0], # Taylor expand out to accel
              [0,      1,        simdt,             0],
              [0,  -fr-kp/simdt,     0, ki/(simdt**2)], # (pos error) / dt^2 puts it in units of accel
              [0,   -simdt,          0,             1]])
# Here inputs are [slope, vel_desired - vel_estimated]
B = np.array([[0,   0],
              [0,   0],
              [-mg, kp/simdt], # (vel error) / dt puts it in units of accel
              [0,   simdt]])
C = np.array([[1, 0, 0, 0]])
P0 = 10*np.eye(A.shape[0]) # See #110 for why this choice of P0
xhat0 = np.zeros(A.shape[0]); xhat0[0] = y[0] # The first estimate is the first seen state. See #110
R = np.array([[r]])
Q = q*dt*np.diag([(0.5 * dt**2)**2, dt**2, 1, (0.5 * dt**2)**2])
u = np.zeros((len(slope), 2))
u[:,0] = slope
u[:,1] = vd

xhat_pre, xhat_post, P_pre, P_post = kalman_filter(y, dt, xhat0, P0, A, Q, C, R, B, u, save_P=True)

xhat_smooth = rts_smooth(dt, A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False)

xhat_robust = convex_smooth(y, A, Q, C, R, B, u, 0, 2)
#print(xhat_robust)

pyplot.figure(figsize=(13,8))
pyplot.plot(t, y, '.', color='blue', markersize=7, label='noisy data')
pyplot.plot(t, x, '--', color='black', linewidth=5, label='true x')
#pyplot.plot(t, xhat_post[:,0], '-', color='red', linewidth=2, label='Kalman estimate')
pyplot.plot(t, xhat_smooth[:,0], '-', color='green', linewidth=2, label='RTS estimate')
pyplot.plot(t, xhat_robust[:,0], '--', color='orchid', linewidth=2.5, label='Robust estimate')
pyplot.xlabel("Time", fontsize=24)
pyplot.ylabel("Position", fontsize=24)
pyplot.title("Cruise Control with Robust Model", fontsize=24)
pyplot.legend(loc='lower right', bbox_to_anchor=(0.85, 0), fontsize=18)
pyplot.tick_params(axis='x', labelsize=15)
pyplot.tick_params(axis='y', labelsize=15)
#pyplot.text(0.05, 0.95, f"Kalman RMSE = {evaluate.rmse(x, xhat_post[:,0]):.3g}", color="red", transform=pyplot.gca().transAxes, fontsize=18, verticalalignment='top')
pyplot.text(0.05, 0.95, f"RTS RMSE = {evaluate.rmse(x, xhat_smooth[:,0]):.3g}", color="green", transform=pyplot.gca().transAxes, fontsize=18, verticalalignment='top')
pyplot.text(0.05, 0.90, f"Robust RMSE = {evaluate.rmse(x, xhat_robust[:,0]):.3g}", color="orchid", transform=pyplot.gca().transAxes, fontsize=18, verticalalignment='top')
pyplot.tight_layout()
#pyplot.show()
pyplot.savefig('cruise_control_robust.png')





