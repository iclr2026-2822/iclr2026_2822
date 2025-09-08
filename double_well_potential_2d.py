"""
2D double-well potential system definitions and trajectory simulator.
Contains potential, gradient, and a function to simulate the SDE trajectory.
"""
import numpy as np
from tqdm import tqdm  # Import tqdm for progress tracking

def potential(x, y):
    """
    Compute the 2D double-well potential V(x, y) = (x^2 - 1)^2 + y^2.
    """
    return (x**2 - 1)**2 + y**2


def gradient(x, y):
    """
    Compute gradient of the potential:
    dV/dx = 4*x^3 - 4*x,  dV/dy = 2*y
    """
    dVdx = 4 * x**3 - 4 * x
    dVdy = 2 * y
    return dVdx, dVdy


def simulate_trajectory(x0, y0, T, h=1e-4, n_steps=100, sigma=1.09):
    """
    Simulate a single SDE trajectory on the double-well landscape.
    Returns data_matrix of shape (1, T+1, 2) and the lag time.
    
    Args:
        x0, y0: Initial position
        T: Number of evaluation steps
        h: Integration time step
        n_steps: Number of integration steps between evaluations
        sigma: Noise intensity
        
    Returns:
        Tuple of (data_matrix, lag_time)
    """
    lag_time = h * n_steps
    print('Lag time:', lag_time)

    data_matrix = np.zeros((1, T + 1, 2))
    x, y = x0, y0
    
    # Add tqdm progress bar to track the trajectory generation
    for j in tqdm(range(T + 1), desc="Generating trajectory", unit="steps"):
        data_matrix[0, j, 0] = x
        data_matrix[0, j, 1] = y
        for _ in range(n_steps):
            dVdx, dVdy = gradient(x, y)
            x += -dVdx * h + sigma * np.sqrt(h) * np.random.normal()
            y += -dVdy * h + sigma * np.sqrt(h) * np.random.normal()
            
    return data_matrix, lag_time