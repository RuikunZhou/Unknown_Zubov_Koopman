from dreal import *
import numpy as np
import sympy as sp

def get_x_bound(x, xlim): 
    bounds_conditions = []
    for i in range(len(xlim)):
        lower_bound = x[i] >= xlim[i][0]
        upper_bound = x[i] <= xlim[i][1]
        bounds_conditions.append(logical_and(lower_bound, upper_bound))
    all_bounds = logical_and(*bounds_conditions)
    return all_bounds

def compute_Lipschitz(x, xlim, f, config, Lip=1, step=0.1):
    jacobian = []
    for f_component in f:
        row = [f_component.Differentiate(x_var) for x_var in x]
        jacobian.append(row)

    # Calculate the sum of squares of all partial derivatives
    derivative_sum = Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            derivative_sum += jacobian[i][j]**2

    x_bound = get_x_bound(x, xlim)
    condition = logical_imply(x_bound, derivative_sum <= Lip)

    violation = CheckSatisfiability(logical_not(condition), config)

    while violation:
        Lip += step
        condition = logical_imply(x_bound, derivative_sum <= Lip)
        violation = CheckSatisfiability(logical_not(condition), config)
    return np.sqrt(Lip)

def compute_dVdx(x, V, xlim, config, M=0.1, step=0.1):
    derivative_of_V = Expression(0)
    x_bound = get_x_bound(x, xlim)
    for i in range(len(x)):
        derivative_of_V += V.Differentiate(x[i])*V.Differentiate(x[i])
  
    condition = logical_imply(x_bound, derivative_of_V <= M)

    violation = CheckSatisfiability(logical_not(condition),config)
    while violation:
        M += step
        condition = logical_imply(x_bound, derivative_of_V <= M)
        violation = CheckSatisfiability(logical_not(condition),config)
        
    return np.sqrt(M)


def compute_Lipschitz_V(x, V, level, f, domain, config, Lip=1, step=0.1):
    jacobian = []
    for f_component in f:
        row = [f_component.Differentiate(x_var) for x_var in x]
        jacobian.append(row)

    # Calculate the sum of squares of all partial derivatives
    derivative_sum = Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            derivative_sum += jacobian[i][j]**2

    level = Expression(level)
    all_bounds = get_x_bound(x, domain)
    x_bound = logical_and(all_bounds, V <= level)
    condition = logical_imply(x_bound, derivative_sum <= Lip)

    violation = CheckSatisfiability(logical_not(condition), config)
    while violation:
        Lip += step
        condition = logical_imply(x_bound, derivative_sum <= Lip)
        violation = CheckSatisfiability(logical_not(condition), config)
        
    return np.sqrt(Lip)

def compute_dVdx_V(x, V, level, domain, config, M=0.1, step=0.1):
    derivative_of_V = Expression(0)
    for i in range(len(x)):
        derivative_of_V += V.Differentiate(x[i])*V.Differentiate(x[i])
  
    all_bounds = get_x_bound(x, domain)
    x_bound = logical_and(all_bounds, V <= level)
    condition = logical_imply(x_bound, derivative_of_V <= M)

    violation = CheckSatisfiability(logical_not(condition),config) 
    while violation:
        M += step
        condition = logical_imply(x_bound, derivative_of_V <= M)
        violation = CheckSatisfiability(logical_not(condition),config)
        
    return np.sqrt(M)

# reture the quadratic Lyapunov function with P
def V_quadratic(x, P):
    V = Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            V += x[i] * P[i][j] * x[j]
    return V

def verify_V_updated(x, V, V_quad, level, c1_P, f, domain, epsilon, config):
    lie_derivative_of_V = Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])

    level = Expression(level)
    all_bounds = get_x_bound(x, domain)
    V_bound = logical_and(c1_P <= V_quad, V <= level)
    x_bound = logical_and(all_bounds, V_bound)
    condition = logical_imply(x_bound, lie_derivative_of_V <= epsilon)
    
    return CheckSatisfiability(logical_not(condition), config)

# Convert dReal expressions to SymPy expressions
def convert_to_sympy(dreal_expr, dreal_vars, sympy_vars):
    expr_str = str(dreal_expr)
    for dreal_var, sympy_var in zip(dreal_vars, sympy_vars):
        expr_str = expr_str.replace(str(dreal_var), str(sympy_var))
    return sp.sympify(expr_str)
