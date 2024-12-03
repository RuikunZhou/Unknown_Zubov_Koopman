import numpy as np
import time
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import utils
import os

np.random.seed(42)

system = 'Van_der_Pol'
dim = 2
M = 200
m_monomial = 8
n_monomial = 8

x_lim = 1.2

span = 5
t_span = [0, span]
frequency = 50
NN = frequency*span + 1
t_eval=np.linspace(0, span, NN)
t_data=np.linspace(0, span, NN) 

x_gauss, w_gauss = leggauss(NN)  # Get Gauss-Legendre nodes and weights for standard interval [-1, 1]

x_gauss_transformed = 0.5 * (x_gauss + 1) * (span - 0) + 0

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_SampleData_samples_{M}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_FlowData_{frequency}_samples_{M}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)

if __name__ == "__main__": 
    y_data = np.zeros((M, NN))
    interpolated_values = np.zeros((M, NN))
    
    YL = np.zeros((M, m_monomial*n_monomial))
    YR = np.zeros((M, m_monomial*n_monomial))

    # Given a list of mu values, save the error for each mu value
    mu_values = np.concatenate((np.array([0.02]), np.arange(0.25, 4, 0.25), np.arange(4, 21, 1)))
    lamda = 1e8 

    print('This is for frequency: ', frequency)
    error_values = []
    for mu in mu_values:
        print('Processing mu = ', mu)
        tic2 = time.time()
        exp_term = mu**2 * np.exp(-mu * t_data) 
        j = 0
        for n in range (n_monomial):
            for m in range(m_monomial):
                # print('processing basis i=', j)
                eta_flow = np.power(flow_data[:,0,:], m) * np.power(flow_data[:,1,:], n)  
                eta_sample = np.power(sample[:,0], m) * np.power(sample[:,1], n) 
                
                y_data = exp_term * eta_flow  
                
                # Perform interpolation in a vectorized manner
                interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
                interpolated_values = interpolator(x_gauss_transformed)  

                # Compute integrals
                inte = np.dot(interpolated_values, w_gauss) * 0.5 * span 
                
                # Update YL, YR
                YL[:, j] = inte / (mu**2) * (lamda - mu) + eta_sample
                YR[:, j] = inte / mu * lamda- lamda* eta_sample
                j+=1
        print('processing time = ', time.time()-tic2)
        ##############################################################################
        
        pinv_L = np.linalg.pinv(YL)
        L_update = pinv_L @ YR 

        logfree_weights = np.vstack((L_update[:,1], L_update[:,m_monomial]))

        # Define the path for the subfolder
        subfolder = "results"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        weight_file = os.path.join(subfolder, f'{system}_logfree_weights_M_{M}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy')        

        np.save(weight_file, logfree_weights)

        # save L_update
        L_update_file = os.path.join(subfolder, f'{system}_L_M_{M}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy')
        np.save(L_update_file, L_update)

        ##############################################################################
        # check if the exact weights file exists or not, if yes, load the weights
        try:
            coeff_exact = np.load(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}.npy')
        except:
            coeff_exact = utils.extract_coefficients(m_monomial, n_monomial)
            np.save(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}.npy', coeff_exact)

        logfree_error = np.linalg.norm(coeff_exact - logfree_weights)

        # calculate the root mean square error 
        rmse = logfree_error/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])

        print(f'RMSE for frequency {frequency} with mu={mu}: {rmse: .2e}')
        error_values.append([mu, rmse])
        print('#'*40)

    error_values = np.array(error_values)
    # save the error values
    np.save(f'{system}_error_values_f_{frequency}.npy', error_values)
    # plotting scatter plot
    plt.scatter(mu_values, error_values[:,1])
    plt.xlabel(f'values $\mu$')
    plt.ylabel('RMSE')
    plt.yscale('log')  
    plt.title(f'RMSE vs Mu for f={frequency} with (m={m_monomial}, n={n_monomial}), $\lambda$={lamda:.1e}')
    plt.show()
    # save the plot
    plt.savefig(f'{system}_RMSE_vs_Mu({lamda:.1e})_{frequency} with (m={m_monomial}, n={n_monomial}).png')
    print('#'*50)
