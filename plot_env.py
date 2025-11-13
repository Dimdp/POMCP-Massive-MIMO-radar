import matplotlib.pyplot as plt
import numpy as np


def plot_function(env, radar, sigma2_c, Tmax):
    for k in range(20):
        x, nu_trajectory = env.generate_trajectory(Tmax)
        plt.plot(x[:,0], x[:,2])
        plt.xlabel('x(t) ')
        plt.ylabel('y(t) ')
    plt.savefig('./results/plots/trajcetory_samples.png')
    plt.close()

    
    for k in range(20):
        x, nu_trajectory = env.generate_trajectory(Tmax)
        snr = radar.get_snr_from_range(np.sqrt(x[:,0]**2 + x[:,2]**2), sigma2_c)
        plt.plot(snr)
    plt.savefig('./results/plots/snr_trajectories.png')
    plt.close()


    for k in range(20):
        x, nu_trajectory = env.generate_trajectory(Tmax)
        plt.plot(np.sqrt(x[:,0]**2 + x[:,2]**2))
    plt.savefig('./results/plots/ranges.png')
    plt.close()


    for k in range(20):
        x, nu_trajectory = env.generate_trajectory(Tmax)
        for t in range(1, len(nu_trajectory)):
            if nu_trajectory[t] - nu_trajectory[t-1] > 1:
                print('Trajectory jump', nu_trajectory[t] - nu_trajectory[t-1])
        plt.plot(nu_trajectory)
    
    print(nu_trajectory)
    plt.savefig('./results/plots/angle_trajectory.png')
    plt.close()
    
    for k in range(20):
        x, nu_trajectory = env.generate_trajectory(Tmax)
        r_alpha = []
        i_alpha = []
        for t in range(Tmax):
            _, real_alpha, imag_alpha, ph = radar.x_y_to_alpha(x[t,0], x[t,2])
            r_alpha.append(real_alpha)
            i_alpha.append(imag_alpha)
        
        module = np.sqrt(np.array(r_alpha)**2 + np.array(i_alpha)**2)
        plt.plot(module)
        print('np max = ', np.max(module))
        print('np min = ', np.min(module))
    plt.savefig('./results/plots/module_alpha_trajectory.png')
    plt.close()
    
    print('Plots saved ...')