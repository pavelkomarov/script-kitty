import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter, savgol_coeffs
from pynumdiff.utils.utility import friedrichs_kernel, gaussian_kernel

N = 15

# IIR iterated FD filter
#b = np.array([1, 1, -1, -1])/4
#a = [1, -1]

b = savgol_coeffs(N, 5)
a = 1
# b2 = savgol_coeffs(N, 5, deriv=1)

#w2, h2 = freqz(b2, worN=8000, include_nyquist=True)
w, h = freqz(b, a, worN=8000, include_nyquist=True)  # worN controls the resolution of the response

# Plot the magnitude response (in dB)
# fig, ax = plt.subplots(2, 2, figsize=(18,5), constrained_layout=True)
# for i,(b,a) in enumerate([(np.ones(N)/N, 1), (gaussian_kernel(N), 1), (friedrichs_kernel(N), 1), butter(4, 0.3)]):
# 	w, h = freqz(b, a, worN=8000, include_nyquist=True)  # worN controls the resolution of the response
# 	ax[i//2,i%2].plot(w / np.pi, 20 * np.log10(np.abs(h)))
# 	ax[i//2,i%2].set_ylim(-80, 1)
# 	ax[i//2,i%2].grid(True)
# 	ax[i//2,i%2].tick_params(axis='y', labelsize=13)
# 	ax[i//2,i%2].tick_params(axis='x', labelsize=14)

# ax[0,0].set_title(fr"Frequency response of Moving Average filter with width {N}", fontsize=16)
# ax[0,1].set_title(fr"Frequency response of Gaussian kernel filter with width {N}", fontsize=16)
# ax[1,0].set_title(fr"Frequency response of Friedrichs kernel filter with width {N}", fontsize=16)
# ax[1,1].set_title(fr"Frequency response of $4^\text{{th}}$-order Butterworth filter with cutoff 0.3 of Nyquist", fontsize=15)
# ax[0,0].set_xticklabels([])
# ax[0,1].set_xticklabels([])
# ax[0,1].set_yticklabels([])
# ax[1,1].set_yticklabels([])
# ax[0,0].set_ylabel("Magnitude (dB)", fontsize=16)
# ax[1,0].set_ylabel("Magnitude (dB)", fontsize=16)
# ax[1,0].set_xlabel("Frequency as a fraction of the Nyquist rate", fontsize=16)
# ax[1,1].set_xlabel("Frequency as a fraction of the Nyquist rate", fontsize=16)

plt.figure(figsize=(10,3))
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
#plt.title(fr"Frequency response of iterated $2^\text{{nd}}$-order FD with 1 round of trapezoidal integration", fontsize=15)
plt.title("Frequency response of Savitzky-Golay filter with width 15 and polynomial degree 5", fontsize=15)

plt.grid(True)
plt.ylim(-80,1)
plt.ylabel("Magnitude (dB)", fontsize=15)
plt.xlabel("Frequency as a fraction of the Nyquist rate", fontsize=15)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('savgol_freq_response.png')
#plt.savefig('iterated_fd_freq_response.png')
#plt.savefig('butter_freq_response.png')
#plt.savefig('gaussian_freq_response.png')
#plt.savefig('friedrichs_freq_response.png')
#plt.savefig('freq_responses.png')
