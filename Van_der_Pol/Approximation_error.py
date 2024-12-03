import numpy as np
import time
import os

np.random.seed(42)

def f_true(x):
    y = np.zeros_like(x)
    y[:,0] = - x[:,1]
    y[:,1] = x[:,0] + (x[:,0]**2-1)*x[:,1]
    return y

# Define a function to evaluate the basis functions directly without using sympy
def evaluate_basis(dataset_test, m_monomial, n_monomial):
    basis_values = []
    for n in range(n_monomial):
        for m in range(m_monomial):
            basis_values.append((dataset_test[:, 0]**m) * (dataset_test[:, 1]**n))
    return np.column_stack(basis_values)

def evaluate_dxdt(dataset_test, weights):
    Basis = evaluate_basis(dataset_test, m_monomial, n_monomial)
    dxdt = Basis @ weights.T
    return dxdt

dim = 2
system = 'Van_der_Pol'
xlim = 1.2
M = 10
m_monomial = 3
n_monomial = 3
mu = 3.
lamda = 1e8
frequency = 50

# load the weights for the logfree method from results folder 
subfolder = "results"
logfree_file = os.path.join(subfolder, f'{system}_logfree_weights_M_{M}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy')
Logfree_M = np.load(logfree_file) 

# evaluate the errors for all the points in the test dataset
delta = 3e-4
domain = [[-2., 2.], [-2., 2.]]
N_test = int(2 * xlim / delta) + 1
x1_sample = np.linspace(domain[0][0], domain[0][1], N_test, dtype=float)
x2_sample = np.linspace(domain[1][0], domain[1][1], N_test, dtype=float)
x1_test, x2_test = np.meshgrid(x1_sample, x2_sample)
dataset_test = np.vstack([x1_test.ravel(), x2_test.ravel()]).T

# Calculate the true and approximate values
y_true = f_true(dataset_test)
tic = time.time()
y_logfree = evaluate_dxdt(dataset_test, Logfree_M)
toc = time.time()
print("Time for evaluation: ", toc - tic)

# maximum of loss
loss_logfree = np.max(np.abs(y_true - y_logfree))
print('This is for mu =', mu)
print(f"Maximum loss for logfree: {loss_logfree:.2e}")
compute_delta = np.sqrt(dim * delta**2/2)
print(f"Delta: {compute_delta:.2e}")