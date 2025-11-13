import numpy as np
import gc
import random
import ray
import csv

#ray.init(num_cpus = 64)

import multiprocessing as mp
print('Number of cpus = ', mp.cpu_count())

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import time

from utils.env import Environement, ParticleFilter, Radar
from utils.env_helper import *
from pomcp.pomcp import POMCP
from plot_env import plot_function

np.set_printoptions(suppress=True)

def z(r, phi):
    return r * np.exp(1j * 2 * np.pi * phi)


############# noise parameters ################
p = [z(0.5, -0.4), z(0.6,-0.2), z(0.7, 0.0),
     z(0.4, 0.1), z(0.5, 0.3), z(0.6, 0.35)]

sigma2_c = 1; lambda_c = 2 
###############################################

############ radar parameters #################
n_t = 100
n_r = n_t
nu_bins_number = n_t
###############################################



# initial state of the target
dt = 1.0
sigma_s = 0.03
state0 = np.array([60 , 0.2, -60 , 0.2])
a = 60 # dB value for the parameter in the alpha parameter in equation |alpha|² = a - 40 * np.log10(R)
 
x_y_cell_size = 1000/1000
x_y_step = 250/1000

v_cell_size = 0.35
v_step = 2*v_cell_size/12




########### POMCP parameters ####################
depth = 2 # depth of the tree
rollout_depth = 70
n_particles = int( (2*x_y_cell_size/ x_y_step)**2 * (2*v_cell_size/ v_step)**2 )
n_simulations = n_particles if n_particles <= 10000 else 10000 # number of simulations in the tree
n_random_samples = 0/100

cc = np.sqrt(2) # exploration-exploitation hyper-parameter
#################################################

n_mc_runs = 250 # Monte Carlo runs
Tmax = 100 # length of each Monte carlo run
###############################################

print('Number of Simulations = ', n_simulations)
print('Number of Particles = ', n_particles)
print('Depth of the tree = ', depth)
print('Number of Monte Carlo runs = ', n_mc_runs)
print('Length of Monte Carlo runs = ', Tmax)


RADAR = Radar(a, n_r= n_r, n_t= n_t, nu_bins_number= nu_bins_number)

ENV = Environement(RADAR, p= p, sigma2_c= sigma2_c, lambda_c= lambda_c, dt=dt, sigma_s=sigma_s, n_particles= n_particles)
ENV.initialize_state(state0)
PF = ParticleFilter(n_particles= n_particles)
ORACLE = ParticleFilter(n_particles= n_particles)

plot_function(ENV, RADAR, sigma2_c, Tmax)

del ENV
gc.collect()
print('Simulations starts ...')





@ray.remote
def run_simulation(iteration, RADAR, PF, ORACLE, 
                   alpha_min=0.0, alpha_max=1.0):
    
    np.set_printoptions(suppress=True)
    
    start_time = time.time()
    ENV = Environement(RADAR, p =p, sigma2_c =sigma2_c, lambda_c =lambda_c, dt=dt, sigma_s=sigma_s,n_particles = n_particles)
    ENV.initialize_state(state0)
    obs_old = 0
    counter = 1
    k = 0
    while obs_old == 0:
        next_state, nu_bin, observations, estimations, snr = ENV.take_action(0) 
        action_0 = np.argmax(estimations['wald_statistic'][0,:]) + 1
        next_state, nu_bin, observations, estimations, snr = ENV.take_action(action_0) 
        obs_old = estimations['det_arrays'][0, action_0 - 1]
        k = k + 1
        if k > counter:
            return -1
    
    sigma_alpha = estimations['sigma_alpha'][0, action_0-1]
    ENV.add_configs(sigma_alpha, alpha_min, alpha_max)
    x0 ,y0, belief_pomcp = ENV.estimations_to_x_y(estimations, x_y_step, x_y_cell_size, v_cell_size, v_step)
    print('@@@@ x0 = ', x0 , 'y0 = ', y0 )
    print('@@@@ sigma_alpha = ', sigma_alpha)
                    
    SOLVER = POMCP(ENV)
    SOLVER.add_configs(initial_belief= belief_pomcp, max_particles = n_particles, C=cc, reinvigorated_particles_ratio=0.0)

    belief_pf = belief_pomcp.copy()
    belief_oracle = belief_pomcp.copy()
    print('@@@@ Initial belief size = ', len(belief_pomcp))
    print('@@@@ Initial belief mean = ', np.mean(belief_pomcp, axis = 0))
    print('@@@@ Current state = ', ENV.curr_state)
    
    gt_states = [ENV.curr_state]
    pomcp_states = [np.mean(belief_pomcp, axis = 0)]
    pf_states = [np.mean(belief_pf, axis = 0)]
    oracle_states = [np.mean(belief_oracle, axis = 0)]
    
    pd_oracle = []
    pd_pomcp = []
    pd_pf = []
    pd_orthogonal = []
    stop = False
    
    for t in range(Tmax):
        start_time_in_loop = time.time()
        SOLVER.solve(depth, n_simulations)  
        action_pomcp = SOLVER.get_action(belief_pomcp) 
        action_pf = PF.get_action(ENV, belief_pf)

        next_state, nu_bin, observations_index, estimations, snr = ENV.take_action(action_pomcp, action_pf)
        belief_pomcp = SOLVER.update_belief(belief_pomcp, action_pomcp, observations_index[0])
        belief_pf = PF.update_belief(ENV, belief_pf, observations_index[1], action_pf)
        belief_oracle = ORACLE.update_belief(ENV, belief_oracle, observations_index[2], nu_bin + 1)
        
        if (len(belief_pomcp) == 0) or (len(belief_pf) == 0) or (len(belief_oracle) == 0):
            stop = True
            if len(belief_pomcp) == 0:
                print('POMCP Empty')
            if len(belief_pf) == 0:
                print('PF Empty')
            if len(belief_oracle) == 0:
                print('ORACLE Empty')
            break
            
        gt_states.append(ENV.curr_state)
        pomcp_states.append(np.mean(belief_pomcp, axis = 0))
        pf_states.append(np.mean(belief_pf, axis = 0))
        oracle_states.append(np.mean(belief_oracle, axis = 0))

        print('target states = ', np.round(gt_states[-1], 3),    '| target bin = ', 1 + nu_bin)
        print('pomcp  states = ', np.round(pomcp_states[-1], 3), '|  pomcp bin = ', action_pomcp)
        print('pf     states = ', np.round(pf_states[-1], 3),    '|     pf bin = ', action_pf)
        print('oracle states = ', np.round(oracle_states[-1], 3),'| oracle bin = ', 1 + nu_bin)
                
        print('t = ', t , '- time = ', np.round(time.time() - start_time_in_loop, 2))
        print(' - belief size of POMCP, PF, ORACLE = ', len(belief_pomcp), ' , ', len(belief_pf),  ' , ', len(belief_oracle))
        print('----------------------------')
        
        pd_orthogonal.append(estimations['det_arrays'][3, nu_bin])
        pd_oracle.append(estimations['det_arrays'][2, nu_bin])
        pd_pf.append(estimations['det_arrays'][1, nu_bin])
        pd_pomcp.append(estimations['det_arrays'][0, nu_bin])
        
        
    gt_states = np.array(gt_states)
    pomcp_states = np.array(pomcp_states)
    pf_states = np.array(pf_states)
    oracle_states = np.array(oracle_states)

    mse_pomcp = (gt_states[:,0] - pomcp_states[:,0])**2 + (gt_states[:,2] - pomcp_states[:,2])**2
    mse_velocity_pomcp = (gt_states[:,1] - pomcp_states[:,1])**2 + (gt_states[:,3] - pomcp_states[:,3])**2
    
    mse_pf = (gt_states[:,0] - pf_states[:,0])**2 + (gt_states[:,2] - pf_states[:,2])**2
    mse_velocity_pf = (gt_states[:,1] - pf_states[:,1])**2 + (gt_states[:,3] - pf_states[:,3])**2
    
    mse_oracle = (gt_states[:,0] - oracle_states[:,0])**2 + (gt_states[:,2] - oracle_states[:,2])**2
    mse_velocity_oracle = (gt_states[:,1] - oracle_states[:,1])**2 + (gt_states[:,3] - oracle_states[:,3])**2
    
    print('@@@@@@@@ iteration = ', iteration, ' Elapsed time = ', np.round(time.time() - start_time, 2), '@@@@@@@')
    if stop == False:
        return [mse_pomcp, mse_velocity_pomcp, 
                mse_pf, mse_velocity_pf, 
                mse_oracle, mse_velocity_oracle,
                pd_pomcp, pd_pf, pd_oracle, pd_orthogonal]
    
    if stop == True:
        return -1


#run_simulation(0, RADAR, PF, ORACLE, alpha_min=0.0, alpha_max=1.0)

start_time = time.time()

RADAR_ID = ray.put(RADAR)
PF_ID = ray.put(PF)
ORACLE_ID = ray.put(ORACLE)

results_ids = []

for i in range(n_mc_runs):
    results_ids.append(run_simulation.remote(i, RADAR_ID, PF_ID, ORACLE_ID))    
    
results = []
actual_runs = 0
while len(results_ids):
    done_id, not_done_ide = ray.wait(results_ids, num_returns = 1)
    result_per_run = ray.get(done_id[0])
    if result_per_run != -1:
        results.append(result_per_run)
    actual_runs += 1 
        
    results_ids = not_done_ide
    if result_per_run == -1:
        nn = random.sample(list(range(n_mc_runs)), k=1)[0]
        results_ids.append(run_simulation.remote(nn, RADAR_ID, PF_ID, ORACLE_ID))



time_interval = np.array(list(range(1, Tmax + 1)))
rmse_pomcp = np.sqrt(np.mean([x[0] for x in results ], axis = 0))[1:]
rmse_velocity_pomcp = np.sqrt(np.mean([x[1] for x in results ], axis = 0))[1:]

rmse_pf = np.sqrt(np.mean([x[2] for x in results ], axis = 0))[1:]
rmse_velocity_pf = np.sqrt(np.mean([x[3] for x in results ], axis = 0))[1:]

rmse_oracle = np.sqrt(np.mean([x[4] for x in results ], axis = 0))[1:]
rmse_velocity_oracle = np.sqrt(np.mean([x[5] for x in results ], axis = 0))[1:]


pd_pomcp = np.mean([x[6] for x in results ], axis = 0)
pd_pf = np.mean([x[7] for x in results ], axis = 0)
pd_oracle = np.mean([x[8] for x in results ], axis = 0)
pd_orthogonal = np.mean([x[9] for x in results ], axis = 0)


data = np.column_stack((time_interval,
                        pd_pomcp, pd_pf, pd_oracle, pd_orthogonal,
                        rmse_pomcp, rmse_pf, rmse_oracle,
                        rmse_velocity_pomcp, rmse_velocity_pf, rmse_velocity_oracle))


filename = './results/results.csv'

with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    csvwriter.writerow(['Time', 'pd_pomcp', 'pd_pf', 'pd_oracle', 'pd_orthogonal',
                        'rmse_pomcp', 'rmse_pf', 'rmse_oracle',
                        'rmse_velocity_pomcp', 'rmse_velocity_pf', 'rmse_velocity_oracle'])
    
    csvwriter.writerows(data)





fig = plt.figure()
plt.plot(pd_oracle, label = 'Oracle')
plt.plot(pd_pf, label = 'PF')
plt.plot(pd_pomcp, label = 'POMCP')
plt.plot(pd_orthogonal, label = 'Orthogonal')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel("Probability of detection")
plt.title('Probability of detection')
plt.savefig('./results/plots/proba_detection.png')
plt.close(fig)


fig = plt.figure()
plt.semilogy(rmse_oracle, label = 'ORACLE')
plt.semilogy(rmse_pf, label = 'PF')
plt.semilogy(rmse_pomcp, label = 'POMCP')


plt.legend()
plt.xlabel('Time (s)')
plt.ylabel("RMSE (km)")
plt.title('RMSE between the predicted coardinates of each algorithm and ground truth')
plt.savefig('./results/plots/semilogy_mse_coardinates.png')
plt.close(fig)



fig = plt.figure()
plt.plot(rmse_oracle, label = 'ORACLE')
plt.plot(rmse_pf, label = 'PF')
plt.plot(rmse_pomcp, label = 'POMCP')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('RMSE (km)')
plt.title('RMSE between the predicted coardinates of each algorithm and ground truth')
plt.savefig('./results/plots/mse_coardinates.png')
plt.close(fig)



fig = plt.figure()
plt.semilogy(rmse_velocity_oracle, label = 'ORACLE')
plt.semilogy(rmse_velocity_pf, label = 'PF')
plt.semilogy(rmse_velocity_pomcp, label = 'POMCP')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel("RMSE (km/s)")
plt.title('RMSE between the predicted velocities of each algorithm and ground truth')
plt.savefig('./results/plots/semilogy_mse_velocities.png')
plt.close(fig)



fig = plt.figure()
plt.plot(rmse_velocity_oracle, label = 'ORACLE')
plt.plot(rmse_velocity_pf, label = 'PF')
plt.plot(rmse_velocity_pomcp, label = 'POMCP')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('RMSE (km)')
plt.title('RMSE between the predicted velocities of each algorithm and ground truth')
plt.savefig('./results/plots/mse_velocities.png')
plt.close(fig)



print('Done !  ----  Elapsed Time = ', int(time.time() - start_time))
print('Number of actual runs = ', actual_runs)


