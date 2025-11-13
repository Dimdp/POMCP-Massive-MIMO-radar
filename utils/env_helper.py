import numpy as np
from numba import njit


@njit
def element_index(array_vector, element):
    return np.argmin(np.abs(array_vector - element))



@njit
def find_observations_index(observations, set_observations):
    tmp = np.sum(np.abs(np.expand_dims(set_observations , 1) - observations), axis = -1)
    obs_indexes = np.argmin(tmp, axis = 0)
    return obs_indexes




@njit
def get_legal_actions(state, nu_bins_number, nu_array):
    num_actions = 5
    nu = 0.5 * np.sin(np.arctan2(state[2], state[0])) 
    nu_bin = element_index(nu, nu_array)
    legal_actions = np.zeros(num_actions, dtype=np.int16)
    
    legal_actions[0] = (nu_bin - 2) % nu_bins_number + 1
    legal_actions[1] = (nu_bin - 1) % nu_bins_number + 1
    legal_actions[2] = nu_bin % nu_bins_number + 1
    legal_actions[3] = (nu_bin + 1) % nu_bins_number + 1
    legal_actions[4] = (nu_bin + 2) % nu_bins_number + 1
    return legal_actions




@njit
def dynamics(xkm1, dt = 1, sigma_s = 0.01): 
    dt2 = (dt**2)/2
    
    A = np.array([[1.0, dt, 0.0, 0.0], [0.0 ,1.0 ,0.0 ,0.0], [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]])
    G = np.array([[dt2 ,0.0], [dt ,0.0], [0.0, dt2], [0.0, dt]])
    
    xk = A.dot(xkm1) + G.dot(np.random.normal(0, sigma_s, size = 2))
    xk[0] = max(xk[0], 0.0)
    return xk



@njit
def dynamics_vectorized(xkm1, dt = 1 , sigma_s = 0.01): 
    dt2 = (dt**2) / 2
    n_samples = xkm1.shape[0]
    
    A = np.array([[1.0, dt, 0.0, 0.0], 
                  [0.0, 1.0, 0.0, 0.0], 
                  [0.0, 0.0, 1.0, dt], 
                  [0.0, 0.0, 0.0, 1.0]])
    
    G = np.array([[dt2, 0.0], 
                  [dt, 0.0], 
                  [0.0, dt2], 
                  [0.0, dt]])
    
    noise = np.random.normal(0, sigma_s, size=(n_samples, 2))
    xk = A.dot(xkm1.T).T + G.dot(noise.T).T
    xk[:, 0] = np.maximum(xk[:, 0], 0.0)
    return xk

