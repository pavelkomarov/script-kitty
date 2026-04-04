# Numerical Experiments

Self-contained scripts exploring numerical methods, signal processing, and applied math ideas.

---

## gappy.py

**Function reconstruction from sparse samples using Chebyshev bases**

Explores whether a compressed sensing approach — randomly subsampling Chebyshev nodes and solving a linear inverse problem to recover spectral coefficients — can reconstruct a function better than standard methods, particularly in the presence of noise.

The target function `y(x) = e^x * sin(5x)` is sampled at a random subset of 80 Chebyshev nodes. The reconstruction works by building the Gram matrix of the lowest-frequency Chebyshev modes restricted to the sampled points, then solving for the coefficients via a Hermitian linear system.

Two cases are tested:

- **Noiseless:** ~20% subsampling, compared against a full DCT-based Chebyshev expansion.
- **Noisy:** Gaussian noise added, then a bootstrap ensemble (5 repetitions of 50% random subsampling, fitting only the lowest 10 modes, averaged) is compared against simply truncating the DCT coefficients to 10 modes.

**Conclusion:** The random subsampling + averaging approach does not outperform plain spectral low-pass filtering on the full data. Truncating high-frequency modes after a standard DCT is simpler and equally effective. It's essentially a self-contained exploratory research script that tests whether gappy sampling can help with noise suppression in spectral methods.

