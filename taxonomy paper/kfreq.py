import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import dlti, freqz

dt = 1e-2
q = 1000
r = 0.001

I = np.eye(3)
A = np.array([[1, dt, (dt**2)/2], # states are x, x', x"
              [0, 1, dt],
              [0, 0,  1]])
C = np.array([[1, 0, 0]]) # we measure only y = noisy x
R = np.array([[r]])
Q = q*np.array([[(dt**5)/20, (dt**4)/8, (dt**3)/6],
               [(dt**4)/8, (dt**3)/3, (dt**2)/2],
               [(dt**3)/6, (dt**2)/2, dt]]) # uncertainty is around the acceleration but propagates to other dimensions
#Q = np.array([[0, 0, 0], [0, 0, 0], [0, 0, q]]) # old way, not technically process noise
P0 = np.array(100*np.eye(3)) # See #110 for why this choice of P0

P = P0
for i in range(1000):
	P_ = A @ P @ A.T + Q
	K = P_ @ C.T @ np.linalg.inv(C @ P_ @ C.T + R)
	P = (I - K @ C) @ P_

print(P)

# Discrete LTI system, using closed-loop filter matrices
system = dlti((I - K @ C) @ A, K, C, np.array([[0]]), dt=dt)

# Convert to transfer function (assumes single input/output)
transfer_fun = system.to_tf()

b = transfer_fun.num
a = transfer_fun.den

# Frequency response
w, h = freqz(b, a, worN=1024)

# Plot
plt.figure(figsize=(10, 3))
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
#plt.ylim(-80, 1)
plt.title(rf'Frequency response of constant acceleration Kalman Filter with $\Delta t = 0.01$, $q={q}$, $r={r}$', fontsize=14)
plt.xlabel("Frequency as a fraction of the Nyquist rate", fontsize=15)
plt.ylabel("Magnitude (dB)", fontsize=16)
plt.grid(True)
#plt.legend()
plt.tight_layout()
#plt.show()
#plt.savefig('kalman_freq_response.png')

L = P @ A.T @ np.linalg.inv(A @ P @ A.T + Q)

system2 = dlti(L, (I - L @ A), I, np.zeros((3,3)), dt=dt)
transfer_fun2 = system2.to_tf()

b2 = transfer_fun2.num
a2 = transfer_fun2.den

w2, h2 = freqz(b2[0], a2, worN=1024)

# Plot
plt.figure(figsize=(10, 3))
plt.plot(w2 / np.pi, 20 * np.log10(abs(h2)))
#plt.ylim(-80, 1)
plt.title(rf'Frequency response of constant acceleration RTS backward pass with $\Delta t=0.01$, $q={q}$, $r={r}$', fontsize=14)
plt.xlabel("Frequency as a fraction of the Nyquist rate", fontsize=15)
plt.ylabel("Magnitude (dB)", fontsize=15)
plt.grid(True)
#plt.legend()
plt.tight_layout()
#plt.show()

h3 = h * h2

plt.figure(figsize=(10, 3))
plt.plot(w2 / np.pi, 20 * np.log10(abs(h3)))
plt.ylim(-80, 4)
plt.title(rf'Frequency response of constant acceleration RTS Smoothing with $\Delta t=0.01$, $q={q}$, $r={r}$', fontsize=13)
plt.xlabel("Frequency as a fraction of the Nyquist rate", fontsize=14)
plt.ylabel("Magnitude (dB)", fontsize=14)
plt.grid(True)
plt.tick_params(axis='both', labelsize=12)
#plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('rts_freq_response.png')
