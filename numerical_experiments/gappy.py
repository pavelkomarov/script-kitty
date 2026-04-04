# Script to try handling noise by random subsampling and fitting to the basis, then averaging the results,
# essentially a compressed sensing approach. Doesn't work better than simply filtering higher modes.

import numpy as np
import scipy
from numpy.polynomial.chebyshev import Chebyshev
from matplotlib import pyplot

# If we sample regularly, then we can FFT to find out how much of each basis function is present, which is great.
# If we (sub)sample irregularly, then we have to solve a linear inverse problem to see how much of each basis
# function is present.

N = 80
x_n = np.cos(np.arange(N+1) * np.pi / N)

P = [1 if np.random.random() < 0.2 else 0 for n in range(N+1)] # where we're sampling, flattened
r = np.sum(P)
support = np.where(P)[0]
print("r = ", r)

psiT = []
for k in range(N+1): # Choose LOWEST FREQUENCY modes
	c = [0 for i in range(k+1)]
	c[k] = 1
	cheb = Chebyshev(c)
	psi_k = cheb(x_n)
	#psi_k /= np.linalg.norm(psi_k) # normalize so <psi_i, psi_i> = 1, kinda a formality, because the Ax=b will scale everything fine
	psiT.append(psi_k)
psiT_unnormalized = np.array(psiT)
psiT = psiT_unnormalized/np.linalg.norm(psiT_unnormalized, axis=1)[:, None]

pyplot.imshow(psiT[:r], cmap='hot')
pyplot.title(r'first $r$ modes of $\Psi^T$')

M = np.zeros((r, r))
for i in range(r):
	for j in range(r):
		M[i, j] = (psiT[i][support]).dot(psiT[j][support]) # the inner product *only at support*

# S = np.linalg.svd(M).S
# print("condition number", S[0]/S[-1])
#print(M)

pyplot.figure()
pyplot.imshow(M, cmap='hot')
pyplot.title("M")

y = lambda x: np.exp(x)*np.sin(5*x)

y_n = y(x_n)
f = np.array([y_n[support].dot(psiT[i][support]) for i in range(r)])
atilde = scipy.linalg.solve(M, f, assume_a='her')

u = psiT[:r].T.dot(atilde)

Tcoeffs = scipy.fft.dct(y_n, 1)
Tcoeffs[0] /= 2; Tcoeffs[N] /= 2
Tcoeffs /= N
u_cheb = Chebyshev(Tcoeffs)

u_cheb2 = psiT_unnormalized.T.dot(Tcoeffs)

pyplot.figure()
pyplot.plot(np.linspace(-1, 1, 100), y(np.linspace(-1, 1, 100)), label="y")
pyplot.plot(x_n, y_n, 'C0+', label=r"$y_n$")
pyplot.plot(x_n, u_cheb(x_n), 'C4+', label="cheb")
pyplot.plot(x_n, u_cheb2, 'C3+', label="cheb2")
pyplot.plot(x_n, u, 'C1+', label="u")
pyplot.plot(x_n[support], y_n[support], 'C2.', ms=4, label="support")
pyplot.title("noiseless")
pyplot.legend()

y_n_with_noise = y(x_n) + 0.1*np.random.randn(*x_n.shape)

m = 10 # filter by only keeping at most 10 modes
u = []
for repetition in range(5):
	P = [1 if np.random.random() < 0.5 else 0 for n in range(N+1)] # where we're sampling, flattened
	r = np.sum(P)
	support = np.where(P)[0]

	m_ = np.min((m, r)) # filter by only keeping at most 10 modes
	print("repetition", repetition, ", r =", r, ", m_ =", m_)
	M = np.zeros((m_, m_))
	for i in range(m_):
		for j in range(m_):
			M[i, j] = (psiT[i][support]).dot(psiT[j][support]) # the inner product *only at support*

	f = np.array([y_n_with_noise[support].dot(psiT[i][support]) for i in range(m_)])
	atilde = scipy.linalg.solve(M, f, assume_a='her')

	u.append(psiT[:m_].T.dot(atilde))
u = np.array(u)
u = np.mean(u, axis=0)

Tcoeffs = scipy.fft.dct(y_n_with_noise, 1)
Tcoeffs[0] /= 2; Tcoeffs[N] /= 2
Tcoeffs /= N
u_cheb_filtered = Chebyshev(Tcoeffs[:m])

pyplot.figure()
pyplot.plot(np.linspace(-1, 1, 100), y(np.linspace(-1, 1, 100)), label="y")
pyplot.plot(x_n, y_n_with_noise, 'C0+', label="noisy samples")
pyplot.plot(x_n, u_cheb_filtered(x_n), 'k+', label="filtered")
pyplot.plot(x_n, u, 'C1+', label="u")
pyplot.title("with noise")
pyplot.legend()
pyplot.show()
