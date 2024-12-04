import os
import time 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from joblib import Parallel, delayed
import torch


def generate_data(system, n_samples=None, v_max=200, 
                  overwrite=False, n_nodes=32, plot=True, 
                  transform="tanh", omega=None):
    domain = system.domain
    d = len(domain)  

    X_MAX = 1e+6
    eps = 1e-7
    T = 1

    def augmented_dynamics(t, z):
        # dz_ = [func(*z[:-1]) for func in system.f_numpy]
        dz_ = list(system.f_numpy(*z[:-1]))
        if omega is not None: 
            # data generation for Lyapunov equation Dv*f = -omega(x)
            # Makes it 2D with shape (1, len(z) - 1)
            z_tensor = torch.tensor([z[:-1]], dtype=torch.float32)  
            dz = omega(z_tensor).detach().numpy()
            # print("dz: ", dz)
        else: 
            dz = sum([s**2 for s in z[:-1]])
        return dz_ + [dz]

    def get_train_output(x, z, depth=0):
        # print("x: ", x)
        if np.linalg.norm(x) <= eps:
            if omega is not None:
                # data generation for Lyapunov equation
                y = z  # no transform needed
                z_T = z
            elif transform == "exp":
                # y = 1 - np.exp(-40/v_max*z)  # 1-exp(-40) is practically 1
                y = 1 - np.exp(-20/v_max*z) 
                z_T = z                
            else:
                y = np.tanh(20/v_max*z)  # tanh(20) is practically 1
                z_T = z
        elif z > v_max or np.linalg.norm(x) > X_MAX:
            y = 1.0
            z_T = v_max  # the largest recorded value for unstable initial cdts 
        else:
            sol = solve_ivp(lambda t, z: augmented_dynamics(t, z), [0, T], 
                            list(x) + [z], rtol=1e-6, atol=1e-9)
            current_x = np.array([sol.y[i][-1] for i in range(len(x))])
            current_z = sol.y[len(x)][-1]
            y, z_T = get_train_output(current_x, current_z, depth=depth+1)
        return [y, z_T]

    def generate_train_data(x):
        y, z_T = get_train_output(x, 0)
        return [x, y, z_T]

    if not os.path.exists('results'):
        os.makedirs('results')
    t_filename = (f'results/{system.name}_data_{n_samples}_samples'
                  f'_v_max_{v_max}.npy')
    z_values = None
    print('_' * 50)
    print("Generating training data from numerical integration:")
    if os.path.exists(t_filename) and not overwrite:
        print("Data exists. Loading training data...")
        t_data = np.load(t_filename)
        x_train, y_train = t_data[:, :-1], t_data[:, -1]
    else:
        print("Generating new training data...")
        start_time = time.time()    
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) 
                            for dim in domain]).T
        results = Parallel(n_jobs=n_nodes)(
            delayed(generate_train_data)(x) for x in tqdm(x_train)
            )
        x_train = np.array([res[0] for res in results])
        y_train = np.array([res[1] for res in results])
        z_values = np.array([res[2] for res in results])
        print("Saving training data...")
        t_data = np.column_stack((x_train, y_train))
        np.save(t_filename, t_data)
        end_time = time.time()
        print(f"Time taken for generating data: " 
              f"{end_time - start_time} seconds.\n")

    if plot:  
        plt.figure()    

        if d == 1:
            plt.scatter(x_train, np.zeros_like(x_train), c=y_train, 
                        cmap='coolwarm', s=1)  
        else:
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, 
                        cmap='coolwarm', s=1)  

        plt.savefig(
            f"results/{system.name}_data_{n_samples}_samples_v_max_{v_max}.pdf"
            )
        plt.close()

        if z_values is not None: 
            plt.figure()
            plt.scatter(z_values, np.zeros_like(z_values) + 0.5)
            plt.yticks([])
            plt.xlabel('Value')
            plt.xlim([0, v_max])
            plt.title('Z Values Clustering')
            plt.savefig(f'results/{system.name}_data_{n_samples}_samples'
                        f'_v_max_{v_max}_z_values.pdf')
            plt.close()

    return np.column_stack((x_train, y_train))
