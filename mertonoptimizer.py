import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import factorial

def merton_likelihood(returns, mu, sigma, lamb, nu, delta, dt=1/252, trunc_n=5):
    n = np.arange(trunc_n + 1)
    weights = np.exp(-lamb * dt) * (lamb * dt)**n / factorial(n)
    means = mu * dt + n * nu
    variances = sigma**2 * dt + n * delta**2
    stds = np.sqrt(variances)

    likelihood = 0.0
    for r in returns:
        pdfs = norm.pdf(r, loc=means, scale=stds)
        mix_pdf = np.sum(weights * pdfs)
        likelihood += np.log(mix_pdf + 1e-12)
    return -likelihood

# Load data
files = [
    "Round 1/algorithms/data/prices_round_1_day_0.csv",
    "Round 1/algorithms/data/prices_round_1_day_-1.csv",
    "Round 1/algorithms/data/prices_round_1_day_-2.csv"
]

# Combine all into one DataFrame
dfs = [pd.read_csv(f, sep=';') for f in files]
full_df = pd.concat(dfs, ignore_index=True)

# Filter for SQUID_INK only
squid_df = full_df[full_df["product"] == 'SQUID_INK'].reset_index(drop=True)

# Extract mid prices and log returns
mid_prices = squid_df["mid_price"].values
log_returns = np.diff(np.log(mid_prices))  # This is what you'll use for MLE

# Initial guesses and bounds
initial_guess = [0.0, 0.2, 0.5, 0.0, 0.1]  # mu, sigma, lambda, nu, delta
bounds = [(-1, 1), (1e-4, 1), (0, 5), (-0.5, 0.5), (1e-4, 1)]

# Optimization
result = minimize(
    lambda params: merton_likelihood(log_returns, *params),
    x0=initial_guess,
    bounds=bounds,
    method='L-BFGS-B'
)

print("Estimated parameters:")
print(f"mu     = {result.x[0]:.4f}")
print(f"sigma  = {result.x[1]:.4f}")
print(f"lambda = {result.x[2]:.4f}")
print(f"nu     = {result.x[3]:.4f}")
print(f"delta  = {result.x[4]:.4f}")
