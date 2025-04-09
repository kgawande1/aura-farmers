import numpy as np
import types
import matplotlib.pyplot as plt

def get_fair_value_merton(
        T: float,
        mu: float, 
        lamb: float, 
        sigma: float, 
        v: float,
        delta: float, 
        prev_price: float,
        mu_w: float, 
        sigma_w: float,
        prev_w: float,
        n: int = 5,
        m: int = 5):
    
    # n -> number of simulations
    # m -> number of future walk

    kappa = np.exp(v + 0.5 * pow(delta, 2)) - 1
    avg = 0

    for _ in range(n):

        W_T = np.random.normal(0, np.sqrt(T)) # brownian motion

        # Number of jumps (Poisson)
        N_T = np.random.poisson(lamb * T)

        # Sum of jump magnitudes (log-normal in log-space)
        jump_sum = np.sum(np.random.normal(v, delta, size=N_T)) if N_T > 0 else 0.0

        # Combine terms
        drift = (mu - lamb * kappa - 0.5 * sigma**2) * T
        diffusion = sigma * W_T

        log_S = np.log(prev_price) + drift + diffusion + jump_sum
        S_T = np.exp(log_S)
        avg += S_T

    return avg / n

x = get_fair_value_merton(
            T=5,
            mu=0.05149140923172328,
            lamb=0.5,
            sigma=0.028816650240054118,
            v=0.5,
            delta=0.06548459284153497,
            prev_price= 2000,   # FIX: Pass a float instead of the entire price_cache
            mu_w=0.5,
            sigma_w=0.5,
            prev_w=0
)

print(x)