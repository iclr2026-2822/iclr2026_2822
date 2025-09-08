"""
2D Noisy Duffing Oscillator System
"""
import numpy as np
from tqdm import tqdm

def duffing_system(t, state, delta=0.5, alpha=-1, beta=1, gamma=0, omega=0):
    """
    Duffing oscillator system equations.
    dx/dt = y
    dy/dt = -delta*y - alpha*x - beta*x^3 + gamma*cos(omega*t)
    
    Args:
        t: Time
        state: System state [x, y]
        delta: Damping parameter
        alpha: Linear stiffness parameter
        beta: Nonlinear stiffness parameter
        gamma: Forcing amplitude
        omega: Forcing frequency
        
    Returns:
        List [dxdt, dydt] of derivatives
    """
    x, y = state
    dxdt = y
    dydt = -delta*y - alpha*x - beta*(x**3) + gamma*np.cos(omega*t)
    return [dxdt, dydt]

def simulate_trajectory(x0, y0, T, h=1e-4, n_steps=100, sigma=1.2,
                       delta=0.5, alpha=-1, beta=1, gamma=0, omega=0):
    """
    Generate a trajectory for the noisy Duffing oscillator system starting from (x0, y0)
    
    Args:
        x0: Initial x-coordinate (position)
        y0: Initial y-coordinate (velocity)
        T: Total simulation time steps
        h: Integration step size
        n_steps: Number of integration steps between each evaluation
        sigma: Diffusion coefficient (noise strength)
        delta, alpha, beta, gamma, omega: System parameters
        
    Returns:
        Tuple of (data_matrix_single, lag_time)
    """
    lag_time = h * n_steps  # Lag time
    
    print('Lag time: ', lag_time)
    n_eval_single = T
    data_matrix_single = np.zeros((1, n_eval_single+1, 2))
    
    # Generate single long trajectory
    x = x0
    y = y0
    t = 0  # Initialize time
    
    for j in tqdm(range(n_eval_single+1)):
        data_matrix_single[0, j, 0] = x
        data_matrix_single[0, j, 1] = y
        
        for k in range(n_steps):
            # Get derivatives from duffing system
            dxdt, dydt = duffing_system(t, [x, y], delta, alpha, beta, gamma, omega)
            
            # Update state with SDE integration (Euler-Maruyama method)
            x += dxdt * h + sigma * np.sqrt(h) * np.random.normal()
            y += dydt * h + sigma * np.sqrt(h) * np.random.normal()

            # Update time
            t += h
    
    return data_matrix_single, lag_time