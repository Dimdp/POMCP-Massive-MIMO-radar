import random
import numpy as np

from numba import  njit 
from utils.env_helper import *
from utils.radar_helper import *
from utils.noise import generate_noise
from scipy.linalg import sqrtm




class Radar(object):
    def __init__(self, 
                 a = 35,
                 n_r= 100, 
                 n_t= 100, 
                 nu_bins_number= 100, 
                 p_tot= 1,
                 p_fa = 10**(-4)):
        
        self.a = a
        self.n_r = n_r
        self.n_t = n_t
        self.n = self.n_r * self.n_t
        self.nu_bins_number = nu_bins_number
        self.p_tot = p_tot
        self.p_fa = p_fa
        
        self.threshold = -2*np.log(p_fa)
        self.truncation_lag = np.ceil(self.n**(1/4)).astype('int64')
        
        self.nu_array = np.linspace(-0.5, 0.5, self.nu_bins_number)
        self.a_t = np.array([np.exp(1j * 2 * np.pi * np.array(list(range(self.n_t))) * nu) for nu in self.nu_array]).T
        self.a_r = np.array([np.exp(1j * 2 * np.pi * np.array(list(range(self.n_r))) * nu) for nu in self.nu_array]).T
        
        self.hs = {}
        
        for angle_bin in range(self.nu_bins_number + 1):
            if angle_bin == 0 :
                self.hs[angle_bin] = self.get_h_all_bins(angle_bin)
            else:
                self.hs[angle_bin] = self.get_h_all_bins([angle_bin])
    
    def __call__(self, angle_bin):
        return self.hs[angle_bin]
    
    def get_w_matrix(self, angle_bin):
        if angle_bin == 0:
            return np.sqrt(self.p_tot/self.n_t)*np.eye(self.n_t)
        else:
            angle_bins_p = [a - 1 for a in angle_bin]
            A = self.a_t[:, angle_bins_p]
            m, num_targets = A.shape
            A = np.expand_dims(A, -1)
            G = np.zeros((m,m))
            for i in range(num_targets):
                G = G + np.conj(A[:,i,:]).dot((A[:,i,:].T))
                
            tmp = 10**(-10) * np.eye(m)
            G = G + tmp
            G = (G + G.conj().T)/2
            G = (self.p_tot/np.trace(G))*G

            return sqrtm(G)
        
    def get_h_all_bins(self, angle_bins):
        W = self.get_w_matrix(angle_bins)
        h_per_bin = []

        for l in range(self.nu_bins_number):
            h = np.kron(W.T.dot(self.a_t[:,l]), self.a_r[:,l]).astype('complex64')
            norm2_h = np.sum(np.abs(h)**2)
            h_per_bin.append([h, norm2_h])

        return h_per_bin        

    def get_snr_from_range(self, R, sigma2_c):
        return get_snr_from_range(self.a, R, sigma2_c)

    def get_range_from_snr(self, snr, sigma2_c):
        return get_range_from_snr(self.a, snr, sigma2_c)
    
    
    def nu_snr_to_x_y(self, nu, snr, sigma2_c):
        t = np.arcsin(2 * nu)
        R = self.get_range_from_snr(snr, sigma2_c)
        x = R * np.cos(t)
        y = R * np.sin(t)
        return x,y
    
    def x_y_to_nu_snr(self, x, y, sigma2_c, vec=False):
        if vec == False:
            nu = 0.5 * np.sin(np.arctan2(y, x))  
            nu_bin = element_index(self.nu_array , nu)
                
            R = np.sqrt(x**2 + y**2)
            snr = self.get_snr_from_range(R, sigma2_c)
            return nu, nu_bin, snr, R
        else:
            nu = 0.5 * np.sin(np.arctan2(y, x)) 
            nu_bin = np.argmin(
                np.abs(self.nu_array.reshape(-1,1) - nu), axis = 0)
                
            R = np.sqrt(x**2 + y**2)
            snr = self.get_snr_from_range(R, sigma2_c)
            return nu, nu_bin, snr, R
    
    def x_y_to_alpha(self, x, y, vec = False):
        if vec == False:
            nu = 0.5 * np.sin(np.arctan2(y, x))  
            nu_bin = element_index(self.nu_array , nu)
            
            R = np.sqrt(x**2 + y**2)
            received_p = self.a - 40 * np.log10(R)
            alpha_amplitude = 10**(received_p/20)
            
            phi = np.random.uniform(0,1)
            alpha =  alpha_amplitude * np.exp(1j * 2 * np.pi * phi)
            return nu_bin, alpha.real, alpha.imag, phi
        else:
            nu = 0.5 * np.sin(np.arctan2(y, x)) 
            nu_bin = np.argmin(
                np.abs(self.nu_array.reshape(-1,1) - nu), axis = 0)
            
            R = np.sqrt(x**2 + y**2)
            received_p = self.a - 40 * np.log10(R)
            alpha_amplitude = 10**(received_p/20)
            phi = np.random.uniform(0,1, size = len(x))
            alpha =  alpha_amplitude * np.exp(1j * 2 * np.pi * phi)
            return nu_bin, alpha.real, alpha.imag, phi
        
    def x_y_to_nu_alpha(self, x, y, sigma2_c):
        nu, nu_bin, snr, R  = self.x_y_to_nu_snr(x, y, sigma2_c)
        alpha_amplitude = np.sqrt(sigma2_c) * 10**(snr/20)
        
        phi = np.random.uniform(0,1)
        alphas = np.zeros(self.nu_bins_number).astype('complex64')
        alphas[nu_bin] = alpha_amplitude * np.exp(1j * 2 * np.pi * phi)
        return nu, nu_bin, R, snr, alphas
    
    def compute_estimations(self, y, h, norm2_h):
        n, th, tl = self.n, self.threshold, self.truncation_lag
        return compute_estimations(y, h, norm2_h, n, th, tl)

    
    
        
class Environement(object):
    def __init__(self,
                 radar,
                 p = [], 
                 sigma2_c = 1, 
                 lambda_c = 2, 
                 sigma_s = 0.005,
                 dt = 1,
                 n_particles = 500,
                 discount=0.8):
        
        self.radar = radar
        self.p = p
        self.sigma2_c = sigma2_c 
        self.lambda_c = lambda_c
        
        self.sigma_s = sigma_s
        self.dt = dt
        self.n_particles = n_particles
        self.discount = discount       
                
        self.a = self.radar.a
        self.threshold = self.radar.threshold
        
        self.observation_flag = False
    def initialize_state(self, state):
        self.curr_state = state
    
    def add_configs(self, sigma_alpha, alpha_min, alpha_max):
        self.sigma_alpha = sigma_alpha
        self.alphas = np.arange(alpha_min, alpha_max, np.sqrt(3) * sigma_alpha)
        
        alpha_indexes = np.arange(len(self.alphas)).astype('int64')
        obs_indexes = np.array(np.meshgrid(alpha_indexes)).T.reshape(-1, 1).astype('int64')
        self.set_of_observations = np.concatenate([np.array([[-1]]), obs_indexes], axis = 0).astype('int64')
        self.observation_flag = True
        print('@@@@ Number of Observations = ', self.set_of_observations.shape[0])
        
    def estimations_to_x_y(self, estimations, x_y_step, x_y_cell_size, v_cell_size, v_step):
        target_idx = np.argmax(estimations['wald_statistic'][0,:])
        nu = self.radar.nu_array[target_idx]
        mod_alpha = np.abs(estimations['alpha'][0, target_idx])
        t = np.arcsin(2*nu) 
        
        # np.log10(mod_alpha²) = a - 40 np.log10(r)
        r = 10**(-( 20 * np.log10(mod_alpha) - self.a)/40)

        x0 = r * np.cos(t)
        y0 = r * np.sin(t)
        xx = np.arange(x0 - x_y_cell_size , x0 + x_y_cell_size , x_y_step)
        vx = np.arange(- v_cell_size + v_step/2, v_cell_size , v_step)
        yy = np.arange(y0 - x_y_cell_size , y0 + x_y_cell_size , x_y_step)
        vy = np.arange(- v_cell_size + v_step/2, v_cell_size, v_step)
        belief_pomcp = []

        for k in range(len(xx)):
            for j in range(len(vx)):
                for kk in range(len(yy)):
                    for jj in range(len(vy)):
                        belief_pomcp.append(np.array([xx[k], vx[j], yy[kk], vy[jj]]))
        return x0, y0, belief_pomcp
    
    def get_measurements(self, actions, amplitude_array):
        hs = [self.radar(a) for a in actions]

        estimations = {}
        estimations['det_arrays'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs)).astype('int32')
        estimations['wald_statistic'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs))
        estimations['alpha'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs)).astype('complex128')  
        estimations['sigma_c'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs)) 
        estimations['sigma_alpha'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs)) 
        estimations['phase_alpha'] = np.array([np.zeros(self.radar.nu_bins_number)]*len(hs)) 
        
        for l in range(self.radar.nu_bins_number):
            noise = generate_noise(self.p, self.sigma2_c, self.lambda_c, self.radar.n)
            for nh in range(len(hs)):   
                h, norm2_h = hs[nh][l]
                y = amplitude_array[l] * h + noise
                results = self.radar.compute_estimations(y, h, norm2_h)

                estimations['det_arrays'][nh,l] = results[0]
                estimations['wald_statistic'][nh,l] = results[1]
                estimations['alpha'][nh,l] = results[2]
                estimations['sigma_c'][nh,l] = results[3]
                estimations['sigma_alpha'][nh,l] = results[4]
                estimations['phase_alpha'][nh,l] = results[5]
    
        observations = np.zeros((len(hs), 1))
        if self.observation_flag:
            for k in range(len(hs)):
                if actions[k] == 0:
                    idx = np.argmax(estimations['wald_statistic'][k,:])
                else: 
                    idx = actions[k]-1
                if estimations['det_arrays'][k,idx] == 1:
                    alpha = estimations['alpha'][k,idx]
                    
                    module_alpha = np.abs(alpha)
                    print('#### module_hat_alpha = ', module_alpha, '- action = ', actions[k] )
                    
                    alpha_index = np.argmin(np.abs(self.alphas - module_alpha))
                    observations[k] = np.array([alpha_index])
                else:
                    observations[k] = np.array([-1])
        return observations, estimations
    
    def take_action(self, action, action_pf = None): 
        next_state = dynamics(self.curr_state,  self.dt , self.sigma_s)
        self.curr_state = next_state.copy()
        nu, nu_bin, R, snr, alphas = self.radar.x_y_to_nu_alpha(self.curr_state[0], self.curr_state[2], self.sigma2_c)
        
        print('@@@@ True |alpha| = ', np.abs(alphas[nu_bin]))
        print('@@@@ True SNR = ', snr)
        
        actions = [action]
        if action_pf is not None:
            actions.append(action_pf)
            
        actions.append(nu_bin + 1) # oracle
        actions.append(0) # orthogonal waveform
        observations, estimations = self.get_measurements(actions, alphas)
        
        observations_index = [0]*len(actions)
        if self.observation_flag:
            observations_index = find_observations_index(observations, self.set_of_observations)
        
        return next_state, nu_bin, observations_index, estimations, snr
        
    
    def simulate_action(self, state, action, pd_0 = 0.5):
        chosen_bin = action - 1

        next_state = dynamics(state,  self.dt , self.sigma_s)
        nu_bin, alpha_real, alpha_imag, phase = self.radar.x_y_to_alpha(next_state[0], next_state[2])
        
        alpha_real = np.random.normal(alpha_real, self.sigma_alpha/np.sqrt(2))
        alpha_imag = np.random.normal(alpha_imag, self.sigma_alpha/np.sqrt(2))
        alpha_module = np.sqrt(alpha_imag**2 + alpha_real**2)      
        
        decision_stat = 2 * alpha_module**2 / self.sigma_alpha**2
        detection = -1
        if chosen_bin != nu_bin:
            detection = -1
        else:
            if decision_stat >= self.threshold :
                detection = chosen_bin

        observation = np.array([[-1]])
        if detection != -1:
            alpha_index = np.argmin(np.abs(self.alphas - alpha_module))
            observation = np.array([[alpha_index]])
        
        observation_index = find_observations_index(observation, self.set_of_observations)[0]
        reward = chosen_bin == nu_bin
        return next_state, observation_index, reward, 0

    def simulate_action_vectorized(self, state, action, pd_0 = 0.5):
        chosen_bin = action - 1
        
        next_state = dynamics_vectorized(state,  self.dt , self.sigma_s)
        n_samples = next_state.shape[0]
        
        nu_bin, alpha_real, alpha_imag, phase = self.radar.x_y_to_alpha(next_state[:,0], next_state[:,2], vec = True)
        
        alpha_real = np.random.normal(alpha_real, self.sigma_alpha/np.sqrt(2))
        alpha_imag = np.random.normal(alpha_imag,  self.sigma_alpha/np.sqrt(2))
        alpha_module = np.sqrt(alpha_imag**2 + alpha_real**2)      
        
        decision_stat = 2 * alpha_module**2 / self.sigma_alpha**2
        
        not_detected = (chosen_bin != nu_bin)
        seen = decision_stat >= self.threshold
        detected = (chosen_bin == nu_bin) * seen
        not_detected = not_detected + (chosen_bin == nu_bin) * (seen == 0)

        alpha_indexes = np.argmin(np.abs(self.alphas.reshape(-1,1) - alpha_module), axis=0)* detected - np.ones(n_samples) * not_detected
        detections = nu_bin * detected - np.ones(n_samples) * not_detected

        observations = np.array([alpha_indexes]).T   
        observation_index = find_observations_index(observations, self.set_of_observations)
        
        return next_state, observation_index, 0, 0
    
    def generate_trajectory(self, Tmax):        
        x = []
        x.append(self.curr_state.copy())

        for i in range(1, Tmax):
            x.append(dynamics(x[i-1],  self.dt , self.sigma_s))

        angle_bins =  np.zeros(Tmax)
        for k in range(Tmax):
            nu = 0.5*np.sin(np.arctan2(x[k][2], x[k][0]))
            idx = element_index(self.radar.nu_array, nu)
            angle_bins[k] = int(idx)  
                   
        return np.array(x), angle_bins
    
    def get_legal_actions(self, state):
        return get_legal_actions(state, self.radar.nu_bins_number, self.radar.nu_array)
    
    def get_observations_index(self, observations):
        return find_observations_index(observations, self.set_of_observations)
    
    def cost_function(self, action):
        return 0




class ParticleFilter(object):
    def __init__(self, n_particles):
        self.n_particles = n_particles

    def likelihood(self, env, state, observation, action):
        sj, oj, _, _ = env.simulate_action_vectorized(state, action)
        obs_to_keep = np.where(observation == oj)[0]
                
        if len(obs_to_keep) > 0:
            return list(sj[obs_to_keep,:])
        else:
            return []
    
    def update_belief(self, env, old_belief, observation, action, num_trials = 200):
        new_belief = []
        k = 0
        while len(new_belief) < self.n_particles:
            n_samples = self.n_particles - int(len(new_belief))
            states = np.array(random.choices(old_belief, k = n_samples))
            new_belief += self.likelihood(env, states, observation, action)
            k = k + 1
            if k > num_trials:
                break
        print('##### pf particles with generator ', len(new_belief))
        
        if len(new_belief)== 0:
            nn = self.n_particles - len(new_belief)             
            random_samples = random.choices(old_belief, k = nn)
            random_samples = np.array(random_samples)
            new_belief += list(dynamics_vectorized(random_samples,  env.dt , env.sigma_s))
             
        return new_belief
    
    
    def get_action(self, env, belief):
        belief = np.array(belief, dtype=np.float64)
        prediction = np.mean(dynamics_vectorized(belief,  env.dt , env.sigma_s), axis = 0)
        
        theta_pred = np.arctan2(prediction[2],prediction[0])
        nu_pred = 0.5*np.sin(theta_pred)
        
        action_pf = element_index(env.radar.nu_array, nu_pred) + 1
        return action_pf
            














