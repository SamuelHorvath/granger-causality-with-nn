import numpy as np
from scipy.integrate import odeint


def make_var_stationary(beta, radius=0.97):
    """Rescale coefficients of VAR model to make it stable."""
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, delay=0, kernel=None):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = [np.eye(p, dtype=int) for _ in range(lag)]
    beta = [np.eye(p) * beta_value for _ in range(lag)]

    num_nonzero = int(p * sparsity) - 1
    for l in range(lag):
        for i in range(p):
            choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
            choice[choice >= i] += 1
            beta[l][i, choice] = beta_value
            GC[l][i, choice] = 1

    beta = np.hstack(beta)
    beta = make_var_stationary(beta)

    GC = np.array(GC)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag + delay, T + burn_in):
        val = X[:, (t-lag - delay):t - delay]
        if kernel is not None:
            val = np.apply_along_axis(kernel, 0, val)
        X[:, t] = np.dot(beta, val.flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    """Partial derivatives for Lorenz-96 ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
