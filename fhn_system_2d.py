"""
2D FirzHugh- Nagumo System
"""
import numpy as np
from tqdm import tqdm

def fhn_system(t, state, gamma=1, beta= 1, delta= 0.25, epsilon= 0.05, a1= 1/3, b1=0.5, b2=0.1):
    """
    
    FirzHugh-Nagumo system equations.
    dx/dt = (x- a1*x**3-y)
    dy/dt = epsilon * (x + b1 - b2 * y)
    
    Args:
        t: Time
        state: System state [x, y]
        gamma, beta, delta, epsilon, a1, b1, b2- system parameters 
        
    Returns:
        List [dxdt, dydt] of derivatives
    """
    
    x, y = state
    dxdt = (x- a1*x**3-y)
    dydt = epsilon * (x + b1 - b2 * y)
    return [dxdt, dydt]

def simulate_trajectory(x0, y0, T, h=1e-4, n_steps=100, gamma=1,  beta= 1, delta= 0.25, 
                        epsilon= 0.05, a1= 1/3, b1=0.5, b2=0.1, DX= 0.2, DY= 0.2):
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
    
    #print('Lag time: ', lag_time)
    n_eval_single = T
    data_matrix_single = np.zeros((1, n_eval_single+1, 2))
    
    # Generate single long trajectory
    x = x0
    y = y0
    t = 0  # Initialize time
    
    for j in (range(n_eval_single+1)):
        data_matrix_single[0, j, 0] = x
        data_matrix_single[0, j, 1] = y
        
        for k in range(n_steps):
            # Get derivatives from duffing system
            dxdt, dydt = fhn_system (t, [x, y], gamma, beta, delta, epsilon, a1, b1, b2)
            dW_x = np.sqrt(h*DX) * np.random.normal()
            dW_y = np.sqrt(h*DY) * np.random.normal()
            # Update state with SDE integration (Euler-Maruyama method)
            x += dxdt * h + dW_x
            y += dydt * h + dW_y

            # Update time
            t += h
    
    return data_matrix_single, lag_time