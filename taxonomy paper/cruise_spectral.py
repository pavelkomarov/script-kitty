import numpy as np
from matplotlib import pyplot
from pynumdiff.utils.simulate import pi_cruise_control
from pynumdiff.utils import utility

dt = 0.01
x, x_truth, dxdt_truth = pi_cruise_control(duration=4, dt=dt, noise_type='normal', noise_parameters=[0,0.1])

t = np.arange(0,dt*len(x), dt)

padding = 100
pre = x[0]*np.ones(padding) # extend the edges
post = x[-1]*np.ones(padding)
x_hat1 = np.hstack((pre, x, post))
kernel = utility.mean_kernel(padding//2)
x_hat2 = utility.convolutional_smoother(x_hat1, kernel) # smooth the edges in
x_hat3 = x_hat2.copy()
x_hat3[padding:-padding] = x # replace middle with original signal

t_hat = np.arange(-padding*dt, dt*(len(x)+padding), dt)
M = len(t_hat)
k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))
filt = np.abs(k) < 40

X = np.fft.fft(x_hat3)
X *= filt
x_hat4 = np.fft.ifft(X)

x_hat5 = np.hstack((x, x[::-1]))
t_hat2 = np.arange(0, dt*len(x)*2, dt)
print(len(x_hat5), len(t_hat2))
M2 = len(t_hat2)
k2 = np.concatenate((np.arange(M2//2 + 1), np.arange(-M2//2 + 1, 0)))
filt2 = np.abs(k2) < 40
X = np.fft.fft(x_hat5)
X *= filt2
x_hat6 = np.fft.ifft(X)


fig, axes = pyplot.subplots(2, 2, figsize=(20,8), constrained_layout=True)

axes[0,1].plot(t, x_truth, 'k--', linewidth=3, label=r"true $x$")
axes[0,1].plot(t_hat, x_hat1, '.', color='blue', zorder=-100, markersize=5)
axes[0,1].tick_params(axis='x', labelsize=15)
axes[0,1].set_yticklabels([])
axes[0,1].text(0.1, 0.92, "(b)", transform=axes[0,1].transAxes, ha='center', va='top', fontsize=18)

axes[1,0].plot(t, x_truth, 'k--', linewidth=3, label=r"true $x$")
axes[1,0].plot(t_hat, x_hat2, '.', color='blue', zorder=-100, markersize=5)
axes[1,0].tick_params(axis='x', labelsize=15)
axes[1,0].set_ylabel("Position", fontsize=18)
axes[1,0].set_xlabel("Time", fontsize=18)
axes[1,0].tick_params(axis='y', labelsize=15)
axes[1,0].text(0.1, 0.92, "(c)", transform=axes[1,0].transAxes, ha='center', va='top', fontsize=18)

axes[0,0].plot(t, x_truth, 'k--', linewidth=3, label=r"true $x$")
axes[0,0].plot(t_hat2, x_hat5, '.', color='blue', zorder=-100, markersize=5)
axes[0,0].plot(t_hat2, x_hat6, c='r')
axes[0,0].tick_params(axis='x', labelsize=15)
axes[0,0].set_ylabel("Position", fontsize=18)
axes[0,0].tick_params(axis='y', labelsize=15)
axes[0,0].text(0.1, 0.92, "(a)", transform=axes[0,0].transAxes, ha='center', va='top', fontsize=18)

axes[1,1].plot(t, x_truth, 'k--', linewidth=3, label=r"true signal")
axes[1,1].plot(t_hat, x_hat3, '.', color='blue', zorder=-100, markersize=5, label=r"noisy data")
axes[1,1].plot(t_hat, x_hat4, c='r', label="low-pass filtered fit")
#axes[1,1].set_xlim((-1.1, 11.1))
axes[1,1].set_xlabel("Time", fontsize=18)
axes[1,1].tick_params(axis='x', labelsize=15)
axes[1,1].set_yticklabels([])
axes[1,1].text(0.1, 0.92, "(d)", transform=axes[1,1].transAxes, ha='center', va='top', fontsize=18)
axes[1,1].legend(loc='lower right', fontsize=12)



# window_size = 40      # number of samples in window
# step = 20             # number of samples to move the window
# order = 3             # polynomial order

# # Loop over data and fit polynomial in each window
# for i in range(0, len(x) - window_size + 1, step):
#     t_win = t[i:i+window_size]
#     x_win = x[i:i+window_size]

#     # Fit polynomial
#     coeffs = np.polyfit(t_win, x_win, deg=order)

#     # Generate fit curve
#     t_fit = np.linspace(t_win[0], t_win[-1], 100)
#     x_fit = np.polyval(coeffs, t_fit)

#     pyplot.plot(t_fit, x_fit, linewidth=3, alpha=1)
#pyplot.tight_layout()
#pyplot.savefig("cruise_control.png")

#pyplot.show()
pyplot.savefig("spectraldiff_hacks.png")
