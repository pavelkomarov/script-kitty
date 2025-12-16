import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Example knot vector (must be non-decreasing)
knots = np.array([0,0,0,0, 3, 5,6, 9, 10,10,10,10])  # cubic B-splines (degree=3)
degree = 3

t_plot = np.linspace(0, 10, 1000) # Domain for plotting

plt.figure(figsize=(16, 4))

for i,k in enumerate(knots):
    plt.axvline(x=k, color='gray', linestyle='--', linewidth=2, label='knots' if i==0 else None)

splines = []
n_bases = len(knots) - degree - 1
for i in range(n_bases):
    # Create coefficient vector: 1 at position i, 0 elsewhere
    c = np.zeros(n_bases)
    c[i] = 1.0
    # Construct the B-spline
    spline = BSpline(knots, c, degree)
    splines.append(spline)
    # Plot
    spl = spline(t_plot)
    if i != 7:
        plt.plot(t_plot[spl > 0], spl[spl > 0], label=f"$B_{i}$", linewidth=2)
    else: 
        plt.plot(t_plot[spl > 0], spl[spl > 0], label=f"$B_{i}$", linewidth=2, color='y')

# s = np.array([spl(t_plot) for spl in splines])
# s = np.sum(s, axis=0) # proving it's a partition of unity
# plt.plot(t_plot, s, 'y')

plt.xlim((-1.02, 10.1))
plt.xticks([])
plt.yticks([0, 1])
plt.xlabel("t", fontsize=15)
plt.ylabel("B(t)", fontsize=15)
plt.title("Cubic B-spline basis functions", fontsize=18)
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig("bsplines.png")
#plt.show()