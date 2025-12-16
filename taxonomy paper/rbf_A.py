import numpy as np
import matplotlib.pyplot as plt

sigma = 0.5
lmbd = 0.1

t = np.linspace(0,5)

t_i, t_j = np.meshgrid(t,t)
r = t_j - t_i # radius
rbf = np.exp(-(r**2) / (2 * sigma**2)) # radial basis function kernel, O(N^2) entries
rbf[np.where(rbf < 1e-4)] = 0
drbfdt = -(r / sigma**2) * rbf # derivative of kernel
rbf_regularized = rbf + lmbd*np.eye(len(t))

print(np.linalg.cond(rbf))
print(np.linalg.cond(rbf_regularized))

# A_centered = rbf - np.mean(rbf, axis=0)  # center columns
# corr = np.corrcoef(A_centered, rowvar=False)
# plt.imshow(corr)
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(18,6))

ax[0].imshow(rbf)
ax[0].set_title(r"RBF $\mathbf{A}$ matrix, with $\sigma = 0.5$", fontsize=25)
ax[1].imshow(rbf_regularized)
ax[1].set_title(r"$\mathbf{A} + \mathbf{I}\lambda$, $\lambda = 0.1$", fontsize=25)
ax[2].imshow(drbfdt)
ax[2].set_title(r"$\mathbf{\dot{A}}$", fontsize=25)

for i in range(3):
	ax[i].set_xticklabels([])
	ax[i].set_yticklabels([])
	ax[i].tick_params(left=False, bottom=False)


fig.tight_layout(pad=5.0)
#plt.savefig("rbf_A.png")
plt.show()

# alpha = np.linalg.solve(rbf_regularized, x) # O(N^3)