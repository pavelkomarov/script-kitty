import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, logistic, laplace, uniform

# Define a range of x values
x = np.linspace(-0.75, 0.75, 1000)

# Define parameters for each distribution
mu, sigma = 0, 0.1  # Normal
#loc_logistic, scale_logistic = 0, 0.06  # Logistic
loc_laplace, scale_laplace = 0, 0.1  # Laplace
loc_uniform, width_uniform = -0.2, 0.4  # Uniform (from -2 to 2)

# Compute the PDFs
pdf_normal = norm.pdf(x, mu, sigma)
#pdf_logistic = logistic.pdf(x, loc_logistic, scale_logistic)
pdf_laplace = laplace.pdf(x, loc_laplace, scale_laplace)
pdf_uniform = uniform.pdf(x, loc_uniform, width_uniform)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(x, pdf_normal, label=r'Normal($\mu=0$, $\sigma=0.1$)', linewidth=2)
#plt.plot(x, pdf_logistic, label='Logistic', linewidth=2)
plt.plot(x, pdf_laplace, label=r'Laplace($\mu=0$, $b=0.1$)', linewidth=2)
plt.plot(x, pdf_uniform, label='Uniform(low=-0.2, high=0.2)', linewidth=2)
plt.xlim((x[0], x[-1]))
plt.title('Noise Distributions', fontsize=18)
plt.xlabel(r'Noise Sample, $\eta$', fontsize=18)
plt.ylabel('Probability Density', fontsize=18)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("noises.png")
#plt.show()