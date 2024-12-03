from dreal import *
import numpy as np
import sympy as sp
import lyznet
import os
import verify_utils
import time

def evaluate_basis_symbolic(vars, m_monomial, n_monomial):
    x1, x2 = vars
    basis_values = []
    for n in range(n_monomial):
        for m in range(m_monomial):
            # Create the polynomial basis expressions with x1 and x2
            basis_values.append((x1**m) * (x2**n))
    return basis_values

# Define a function to evaluate the basis functions directly without using sympy
def evaluate_basis(dataset_test, m_monomial, n_monomial):
    basis_values = []
    for n in range(n_monomial):
        for m in range(m_monomial):
            basis_values.append((dataset_test[:, 0]**m) * (dataset_test[:, 1]**n))
    return np.column_stack(basis_values)

def generate_boundary_points(domain, num_points_per_edge):
    if len(domain) == 1:
        x_min, x_max = domain[0]
        return np.array([[x_min], [x_max]])

    if len(domain) >= 2:
        # Generate points for the first two dimensions
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        x_edge_points = np.linspace(x_min, x_max, num_points_per_edge)
        y_edge_points = np.linspace(y_min, y_max, num_points_per_edge)

        # Generate edge points for the first two dimensions
        edge_points = np.array([[x, y] for x in [x_min, x_max] 
                                for y in y_edge_points] +
                                [[x, y] for x in x_edge_points 
                                for y in [y_min, y_max]])

        if len(domain) == 2:
            return edge_points
        # For other dimensions, fix points at their min and max values
        other_dims_fixed_points = []
        for dim_values in domain[2:]:
            min_val, max_val = dim_values
            for fixed_val in [min_val, max_val]:
                fixed_points = np.full((len(edge_points), len(domain)), 
                                        fixed_val)
                fixed_points[:, :2] = edge_points  
                other_dims_fixed_points.append(fixed_points)

        return np.vstack(other_dims_fixed_points)

dim = 2
system = 'Van_der_Pol'
x_lim = [[-1.2, 1.2], [-1.2, 1.2]]
domain = [[-2.5, 2.5], [-3.5, 3.5]]
M = 100
m_monomial = 8
n_monomial = 8
mu = 2.5
lamda = 1e8
frequency = 50

# load the weights for the logfree method from results folder 
subfolder = "results"
logfree_file = os.path.join(subfolder, f'{system}_logfree_weights_M_{M}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy')
logfree_M = np.load(logfree_file) 
L = np.load(os.path.join(subfolder, f'{system}_L_M_{M}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy'))

# Define dReal variables
dreal_x1 = Variable("x1")
dreal_x2 = Variable("x2")
dreal_vars = [dreal_x1, dreal_x2]

config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-3
config.number_of_jobs = 30

# Generate symbolic basis values
symbolic_basis = evaluate_basis_symbolic(dreal_vars, m_monomial, n_monomial)
f_logfree_dreal = np.dot(logfree_M, symbolic_basis)

# compute Lyapunov function using lyznet
x1, x2 = sp.symbols('x1 x2')
vars = sp.Matrix([x1, x2])
f_logfree_sympy = [verify_utils.convert_to_sympy(expr, dreal_vars, vars) for expr in f_logfree_dreal]
# change it to sympy matrix
f_logfree_sympy = sp.Matrix(f_logfree_sympy)
f_logfree_at_origin = f_logfree_sympy.subs({x1: 0, x2: 0})
f_logfree_sympy = f_logfree_sympy - f_logfree_at_origin

f_true = [-dreal_x2, dreal_x1 - (1 - dreal_x1**2) * dreal_x2]

# compute the Lyapunov function
vdp_system_logfree = lyznet.DynamicalSystem(f_logfree_sympy, domain, system)
true_system = lyznet.DynamicalSystem(f_true, domain, system)

# compute a Lyapunov funciton directly using L
mmu = 0.1
# generate N random pairs of points in the domain
N = 3000
samples = np.array([np.random.uniform(dim[0], dim[1], N) 
                            for dim in domain]).T
omega = np.sum(samples**2, axis=1)
basis = evaluate_basis(samples, m_monomial, n_monomial)

A1 = basis @ L - mmu * omega.reshape(-1, 1) * basis
b1 = -mmu * omega

# zero boundary condition
x0 = np.zeros((1, dim))

A2, b2 = evaluate_basis(x0, m_monomial, n_monomial), np.zeros(x0.shape[0])  

boundary_points = generate_boundary_points(domain, 100)
A3, b3 = evaluate_basis(boundary_points, m_monomial, n_monomial), np.ones(boundary_points.shape[0])  

boundary_weight = 100
lhs = np.vstack((A1, boundary_weight * A2, boundary_weight * A3))
rhs = np.hstack((b1, boundary_weight * b2, boundary_weight * b3))

# # Lyapunov function
# lhs = basis @ L
# rhs = -mmu * omega

beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

# create a new folder and save beta
subfolder = "stability_results"
if not os.path.exists(subfolder):    
    os.makedirs(subfolder)
np.save(os.path.join(subfolder, f'{system}_zubov_poly_beta_{N}samples_M_{M}_f_{frequency}_m_{m_monomial}_n_{n_monomial}_lasso.npy'), beta)

# load beta if needed
# beta = np.load(os.path.join(subfolder, f'{system}_zubov_poly_beta_{N}samples_M_{M}_f_{frequency}_m_{m_monomial}_n_{n_monomial}.npy'))

V_poly_dreal = np.dot(symbolic_basis, beta)

# convert it to sympy
V_poly = verify_utils.convert_to_sympy(V_poly_dreal, dreal_vars, vars)

V_poly_call = sp.lambdify((x1, x2), V_poly, 'numpy')

# compute the epsilon within the domain as the initial value
dVdx_bound = verify_utils.compute_dVdx(dreal_vars, V_poly_dreal, x_lim, config)
print("The bound of dVdx = ", dVdx_bound)

# print the time for computing Lipschitz constant
tic1 = time.time()
Lip_f = verify_utils.compute_Lipschitz(dreal_vars, x_lim, f_true, config, Lip=3)
Lip_f_logfree = verify_utils.compute_Lipschitz(dreal_vars, x_lim, f_logfree_dreal, config, Lip=3)
toc2 = time.time()
print("Lipschitz constant for logfree method: ", Lip_f_logfree)
print("Lipschitz constant for true method: ", Lip_f)
print('Time for computing Lipschitz constant=', toc2-tic1)

# compute the epsilon and verify the conditions
delta = 3e-4 
Delta = np.sqrt(dim * delta**2/2)
print("Delta = ", Delta)
alpha = 4.16e-6
epsilon = dVdx_bound*((Lip_f+Lip_f_logfree)*Delta + alpha)
print("Beta = ", epsilon)

c1_P = lyznet.local_stability_verifier(vdp_system_logfree)
c2_P = lyznet.quadratic_reach_verifier(vdp_system_logfree, c1_P, epsilon=epsilon)
c1_V, c2_V = lyznet.numpy_poly2_verifier(vdp_system_logfree, m_monomial, n_monomial, beta, c2_P, 
                                         epsilon=epsilon)

# update the Lipschitz constants
tic3 = time.time()
Lip_f = verify_utils.compute_Lipschitz_V(dreal_vars, V_poly_dreal, c2_V, f_true, domain, config)
Lip_f_logfree = verify_utils.compute_Lipschitz_V(dreal_vars, V_poly_dreal, c2_V, f_logfree_dreal, domain, config)
toc4 = time.time()
print("Lipschitz constant for logfree method within the level set: ", Lip_f_logfree)
print("Lipschitz constant for true dynamcis within the level set: ", Lip_f)
print('Time for computing Lipschitz constant=', toc4-tic3)
# compute new dVdx bound
dVdx_bound = verify_utils.compute_dVdx_V(dreal_vars, V_poly_dreal, c2_V, x_lim, config)
print("The bound of dVdx within the level set= ", dVdx_bound)

epsilon_updated = dVdx_bound*((Lip_f+Lip_f_logfree)*Delta + alpha)
print("Updated Beta = ", epsilon_updated)

P = vdp_system_logfree.P
V_quad = verify_utils.V_quadratic(dreal_vars, P)

if epsilon_updated < epsilon:
    print('The updated epsilon is smaller than the original epsilon. No need to re-verify the Lyapunov conditions.')
    pass
else:
    results = verify_utils.verify_V_updated(dreal_vars, V_poly_dreal, V_quad, c2_V, c1_P, f_logfree_dreal, domain, epsilon_updated, config)
    # verify the Lyapunov function with the new epsilon if needed
    if results:
        print("The Lyapunov function is verified within the level set for the updated beta.")
    else:
        c1_V, c2_V = lyznet.numpy_poly2_verifier(vdp_system_logfree, m_monomial, n_monomial, beta, c2_P, epsilon=epsilon_updated)
    

def V_neural(x1, x2):
    return np.tanh((0.51641756296157837 + 0.75732171535491943 * np.tanh((-1.6187947988510132 + 2.0125248432159424 * x1 - 0.86828583478927612 * x2)) - 1.6154271364212036 * 
                    np.tanh((-1.0764049291610718 + 0.26035198569297791 * x1 - 0.058430317789316177 * x2)) + 1.2375599145889282 * np.tanh((-0.96464759111404419 - 0.50644028186798096 * x1 + 1.4162489175796509 * x2)) + 
                    0.41873458027839661 * np.tanh((-0.82901746034622192 + 2.5682404041290283 * x1 - 1.2206004858016968 * x2)) - 0.89795422554016113 * np.tanh((0.98988056182861328 + 0.83175277709960938 * x1 + 1.0546237230300903 * x2)) 
                    + 1.0879759788513184 * np.tanh((1.1398535966873169 - 0.2350536435842514 * x1 + 0.075554989278316498 * x2)))) 

c_neural = 0.753

# plot V_direct on the domain with phase portrait
lyznet.plot_V_Koopman(vdp_system_logfree, true_system=true_system, V_list=[V_neural], c_lists=[[c_neural]], 
                      c2_V=c2_V, c2_P=c2_P,
                      poly_model=[m_monomial, n_monomial, beta],
                      phase_portrait=True)