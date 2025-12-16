import jax.numpy as jnp
import numpy as np
from jax import grad
from specderiv import cheb_deriv, fourier_deriv
from matplotlib import pyplot

# autograd setup
def sum_logistic(x):
	return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

derivative_fn = grad(sum_logistic)

# chebyshev setup
N = 50
x = jnp.cos(jnp.arange(N + 1)*jnp.pi/N)*2
y_x = 1/(1 + jnp.exp(-x))

# fourier setup
M = 50
th = jnp.linspace(-2, 2, M, endpoint=False)
y_th = 1/(1 + jnp.exp(-th))
y_th_periodic_ext = jnp.concatenate((y_th, jnp.array([1/(1 + jnp.exp(-2))]), y_th[::-1])) # periodically extend
th_periodic_ext = jnp.linspace(-2, 6, 2*M+1, endpoint=False)

dy_exact_x = jnp.exp(x)/(1 + jnp.exp(x))**2
dy_exact_th = jnp.exp(th)/(1 + jnp.exp(th))**2
dy_autograd_x = derivative_fn(x)
dy_autograd_th = derivative_fn(th)
print(type(np.array(y_x)))
dy_cheb = cheb_deriv(np.array(y_x), np.array(x), 1)
dy_fourier_periodic_ext = fourier_deriv(y_th_periodic_ext, th_periodic_ext, 1)
dy_fourier = dy_fourier_periodic_ext[:M] # grab the first M

print("chebyshev")
print("L2 autograd", jnp.mean((dy_autograd_x - dy_exact_x)**2))
print("L2 cheb", jnp.mean((dy_cheb - dy_exact_x)**2))
print("L1 autograd", jnp.max(jnp.abs(dy_autograd_x - dy_exact_x)))
print("L1 cheb", jnp.max(jnp.abs(dy_cheb - dy_exact_x)))
print("fourier")
print("L2 autograd", jnp.mean((dy_autograd_th - dy_exact_th)**2))
print("L2 fourier", jnp.mean((dy_fourier - dy_exact_th)**2))
print("L1 autograd", jnp.max(jnp.abs(dy_autograd_th - dy_exact_th)))
print("L1 fourier", jnp.max(jnp.abs(dy_fourier - dy_exact_th)))

pyplot.plot(x, dy_autograd_x, label="autograd")
pyplot.plot(x, dy_cheb, label="cheb")
pyplot.legend()
pyplot.figure()
pyplot.plot(th, dy_autograd_th, label="autograd")
pyplot.plot(th, dy_fourier, label="fourier")
pyplot.legend()
pyplot.figure()
pyplot.plot(th_periodic_ext, y_th_periodic_ext)
pyplot.plot(th_periodic_ext, dy_fourier_periodic_ext)
pyplot.show()



