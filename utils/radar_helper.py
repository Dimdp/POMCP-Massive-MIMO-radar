
import numpy as np
from numba import njit


@njit                            
def get_range_from_snr(a, snr, sigma2_c):
    sigma_c = np.sqrt(sigma2_c)
    return 10**((snr - a + 20 *  np.log10(sigma_c))/(- 40))

@njit                           
def get_snr_from_range(a, R, sigma2_c):
    sigma_c = np.sqrt(sigma2_c)
    received_p = a - 40 * np.log10(R)
    return received_p - 20 *  np.log10(sigma_c) 



@njit                            
def compute_estimations(y, h, norm2_h, n, threshold, truncation_lag):
    hat_alpha = np.vdot(h, y)/norm2_h
    hat_c = y - hat_alpha * h
    
    phase_alpha = (np.angle(hat_alpha) + np.pi)/(2*np.pi)
    x = hat_c * np.conj(h)
    xc = np.conj(x)
    B1 = np.sum(np.abs(x)**2)
    B2 = 0
    for m in range(truncation_lag):
        B2 = B2 + 2 * np.sum((x[m+1:] * xc[0:(n-m-1)]).real)
        
    b_est = B1 + B2
    wald_statistic = 2 * (norm2_h**2) * (np.abs(hat_alpha)**2)/ b_est
    wald_statistic = wald_statistic * (wald_statistic > 0)
    detection = np.int32(wald_statistic > threshold)
    
    hat_sigma_c = np.sqrt(np.mean( np.abs( y - hat_alpha*h*detection )**2 ))
    #print('sigma_alpha' , np.sqrt(b_est) / norm2_h)
    sigma_alpha = np.sqrt(b_est) / norm2_h
    return detection, wald_statistic, hat_alpha, hat_sigma_c, sigma_alpha, phase_alpha