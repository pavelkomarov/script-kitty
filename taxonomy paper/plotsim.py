import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors
from pynumdiff.utils.simulate import sine, triangle, pop_dyn, linear_autonomous, pi_cruise_control, lorenz_x

cmap = pyplot.get_cmap('turbo', 6) # Assign a unique color for each simulation
colors = [cmap(i) for i in range(6)]; colors[0] = 'purple'; colors[-1] = 'red'
colors[2] = [x*0.8 for x in colors[2]]; colors[3] = mcolors.to_rgb('gold'); colors[3] = [x*0.9 for x in colors[3]]
colors[4] = [min(1, x*1.2) for x in colors[4]]

dt = 0.01

fig, axes = pyplot.subplots(2, 3, figsize=(18,7), constrained_layout=True)
fig.set_constrained_layout_pads(hspace=0.1)
fig2, axes2 = pyplot.subplots(figsize=(15,6), constrained_layout=True)

for i,(sim,title) in enumerate(zip(
	[pi_cruise_control, sine, triangle, pop_dyn, linear_autonomous, lorenz_x],
	["Cruise Control", "Sum of Sines", "Triangles", "Logistic Growth", "Linear Autonomous", "Lorenz First Dimension"])):
	
	x, x_truth, dxdt_truth = sim(duration=4, dt=dt, noise_type='normal', noise_parameters=[0,0.1])

	X = np.fft.fft(x)
	energy = 20*np.log10(np.abs(X)) + i*12
	freqs = np.fft.fftfreq(len(X), dt)
	energy = energy[freqs >=0]
	freqs = freqs[freqs >= 0]
	
	axes2.plot(freqs, energy, label=title, color=colors[i], alpha=0.7, linewidth=2)
	t = np.arange(0, len(x))*dt

	ax = axes[i//3, i%3]
	ax.plot(t, x_truth, 'k--', linewidth=3, label=r"true $x$")
	ax.plot(t, x, '.', color='blue', zorder=-100, markersize=5, label="noisy data")
	if i//3 == 0: ax.set_xticklabels([])
	#if i%3 != 0: ax.set_yticklabels([])
	#ax.tick_params(axis='x', labelsize=15)
	ax.set_title(title, fontsize=18)
	if i == 5: ax.legend(loc='lower right', fontsize=12)

axes2.tick_params(axis='x', labelsize=15)
#axes2.tick_params(axis='y', labelsize=15)
axes2.set_yticklabels([])
axes2.set_title("Power Spectra for Simulated Data", fontsize=18)
axes2.set_xlabel("Frequency in Hz", fontsize=18)
axes2.set_ylabel("Relative Magnitude in dB", fontsize=18)
axes2.set_ylim(-8, 100)
axes2.grid(True)
#axes2.set_xlim(0, 10)
axes2.legend(fontsize=12, loc='upper right')
#pyplot.tight_layout()
pyplot.show()
fig.savefig("sims.png")
fig2.savefig("power_spectra.png")