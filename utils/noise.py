import numpy as np
from scipy.signal import lfilter


def generate_noise(p = [], sigma2_c = 1, lambda_c = 2, n = 10**4):
    scale = 1/(lambda_c-1)
    rho = np.poly(p) 
    n_trans = 100 * np.ceil(3* np.log(10)/np.abs(np.log(np.max(np.abs(p))))).astype('int64') 
    m = n + n_trans
    
    w = np.sqrt(1/2) * (np.random.normal(size = (m)) + 1j * np.random.normal(size = (m)))
    R = np.random.gamma(lambda_c, scale, size=(m))
    innov = np.sqrt(1/R)*w
    cl = lfilter([1], rho, innov)
    c = cl[n_trans:].T
    sigma2_c_est = np.mean(np.abs(c)**2)
    c_norm = (c/np.sqrt(sigma2_c_est)) * np.sqrt(sigma2_c)
    return c_norm.astype('complex64')
