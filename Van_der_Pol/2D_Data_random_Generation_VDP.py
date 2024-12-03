import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing
import os

system = 'Van_der_Pol'
dim = 2
M = 200
x_lim = 1.2

span = 5
t_span = [0, span]
gen_frequency = 1000
NN = gen_frequency*span + 1
t_eval=np.linspace(0, span, NN)

def ode_function(t, var):
    mu = 1.0
    x1, x2 = var
    return [-x2, x1 - mu * (1 - x1**2) * x2] 

#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval):
    y0 = initial_setup
    solution = solve_ivp(ode_function, t_span, y0, t_eval=t_eval, method='Radau', dense_output=True, atol=1e-10, rtol=1e-7) 
    data0 = solution.y[0]
    data1 = solution.y[1]
    return [[data0, data1]]

def ode_data_generator(initial_setups, t_span, t_eval):
    print('Start solving ODE')
    tic1 = time.time()
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    I = np.stack([results[i][0] for i in range(M)], axis=0)
    # print('ODE solving time = {} sec'.format(time.time()-tic1))
    total_time = time.time() - tic1
    print(f"Total time for data generation: {total_time:.2f} seconds")
    return I


if __name__ == "__main__": 
    
    # randomly sample the initial conditions
    sample = np.random.uniform(-x_lim, x_lim, (M, dim))
    initial_setups = [[*sample[i]] for i in range(M)]
    
    #Use all available CPU cores
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    ##############################################################################
    
    flow_data = ode_data_generator(initial_setups, t_span, t_eval)
    
    # Define the path for the subfolder, and save the data in the subfolder
    subfolder = "data"
    # Check if the folder exists, if not, create it
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # given an array of frequency, the data at the corresponding points are extracted
    frequency = [5, 10, 20, 50, 100]

    # extract data at the corresponding points and save as numpy files
    for i in range(len(frequency)):
        loc = np.arange(0, NN, round((NN-1)/span/frequency[i]))
        filenameY = os.path.join(subfolder, f'{system}_FlowData_{frequency[i]}_samples_{M}_span_{span}_x_{x_lim}.npy')    
        np.save(filenameY, flow_data[:,:,loc])
 
    filenameX = os.path.join(subfolder, f'{system}_SampleData_samples_{M}_span_{span}_x_{x_lim}.npy')
    np.save(filenameX, sample)
 
    ##############################################################################