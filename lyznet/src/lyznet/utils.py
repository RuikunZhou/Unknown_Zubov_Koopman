import sympy
import dreal
import numpy
import torch
import random
import lyznet
import time 
import sys
# from z3 import *


def tik():
    global _start_time
    _start_time = time.time()


def tok():
    global _start_time
    if _start_time is not None:
        elapsed_time = time.time() - _start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds.")
        _start_time = None
    else:
        print("Tik was not called before Tok.")


def is_controllable(A, B):
    n = A.shape[0]
    controllability_matrix = B
    for _ in range(n - 1):
        controllability_matrix = numpy.hstack(
            (controllability_matrix, numpy.linalg.matrix_power(A, _ + 1) @ B)
            )
    # print("controllability_matrix:", controllability_matrix)
    return numpy.linalg.matrix_rank(controllability_matrix) == n


def is_stabilizable(A, B):
    n = A.shape[0]
    eigenvalues, eigenvectors = numpy.linalg.eig(A)
    for i, val in enumerate(eigenvalues):
        if numpy.real(val) >= 0:
            eigenvector = eigenvectors[:, i].reshape(-1, 1)
            if numpy.linalg.matrix_rank(numpy.hstack(
                (B, A @ eigenvector))
            ) < n:
                return False
    return True


def numpy_to_torch(x, f_numpy): 
    N, _ = x.shape  
    result = f_numpy(*x.T.cpu().detach().numpy()).T
    # Check if result is a scalar or has less dimensions than expected
    if result.ndim == 2: 
        # constant functions are somehow lamdified improperly
        result = numpy.tile(result[numpy.newaxis, :, :], (N, 1, 1))
    # print("control input shape: ", result.shape)
    output = torch.tensor(result, dtype=torch.float32).transpose(1, 2)
    # print("control output: ", output)
    return output


def lambdify_matrix_torch(M, syms, xs):
    assert(isinstance(M, sympy.Matrix))
    
    # Lambdify each element for torch
    def eval_element(elem, args):
        func = sympy.lambdify(syms, elem, modules=[torch])
        return func(*args)

    # Assemble the tensor
    rows, cols = M.shape
    output = torch.empty(xs.shape[0], rows, cols, dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            output[:, i, j] = eval_element(M[i, j], xs.T)

    return output


def set_random_seed(seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bisection_glb(Check_bound, low, high, accuracy=1e-4):
    best = None
    # Try 'high' first
    if Check_bound(high) is None:
        return high
    while high - low > accuracy:
        mid = (low + high) / 2
        result = Check_bound(mid)
        if result:
            high = mid
        elif result is None:
            low = mid
            best = mid
            # print("Currently verified level: ", best)
    return best


def bisection_lub(Check_bound, low, high, accuracy=1e-4):
    best = None
    # Try 'low' first
    if Check_bound(low) is None:
        return low
    while high - low > accuracy:
        mid = (low + high) / 2
        result = Check_bound(mid)
        if result:
            low = mid
        elif result is None:
            high = mid
            best = mid
            # print("Currently verified level: ", best)
    return best


def sympy_to_dreal(expr, subs):
    if expr.is_Add:
        return sum(sympy_to_dreal(arg, subs) for arg in expr.args)
    elif expr.is_Mul:
        return sympy.prod(sympy_to_dreal(arg, subs) for arg in expr.args)
    elif expr.is_Pow:
        return sympy_to_dreal(expr.base, subs)**int(expr.exp)
    elif expr.is_Function:
        # Handle the Abs function specifically
        if expr.func.__name__ == "Abs":
            return abs(sympy_to_dreal(expr.args[0], subs))
        else: 
            # handle generic functions
            func = getattr(dreal, expr.func.__name__, None)
            if func is not None:
                return func(*[sympy_to_dreal(arg, subs) for arg in expr.args])
            else:
                raise NotImplementedError(f"Function {expr.func.__name__}"
                                          " is not implemented in dreal")
    elif expr.is_Symbol:
        # Convert single symbols to dreal variables, then to dreal expressions
        if expr in subs:
            return dreal.Expression(subs[expr])
        else:
            raise ValueError(f"Symbol '{expr}' not found in subs dictionary.")        
    elif expr in subs:
        return subs[expr]
    else:
        return dreal.Expression(expr)


def sympy_to_z3(expr, subs):
    """
    Convert a SymPy expression to a Z3 expression.

    Args:
    - expr: A SymPy expression.
    - subs: A dictionary mapping SymPy symbols to Z3 variables.
    """
    if expr.is_Add:
        return sum(sympy_to_z3(arg, subs) for arg in expr.args)
    elif expr.is_Mul:
        return sympy.prod(sympy_to_z3(arg, subs) for arg in expr.args)
    elif expr.is_Pow:
        base = sympy_to_z3(expr.base, subs)
        # Z3 does not support real power, only integer. Handle special cases.
        if expr.exp.is_Integer:
            return base**int(expr.exp)
        else:
            raise NotImplementedError("Z3 only supports integer exponents.")
    elif expr.is_Function:
        # Handle specific functions, like Abs, Sin, Cos, etc.
        if expr.func.__name__ == "Abs":
            return z3.Abs(sympy_to_z3(expr.args[0], subs))
        else:
            # Attempt to find a matching Z3 function
            func = getattr(z3, expr.func.__name__, None)
            if func is not None:
                return func(*[sympy_to_z3(arg, subs) for arg in expr.args])
            else:
                raise NotImplementedError(f"Function '{expr.func.__name__}' is not implemented in Z3")
    elif expr.is_Symbol:
        # Convert single symbols to Z3 variables
        if expr in subs:
            return subs[expr]
        else:
            raise ValueError(f"Symbol '{expr}' not found in subs dictionary.")
    elif expr.is_Number:
        # Convert SymPy numbers to Z3 numbers
        return RealVal(expr)
    else:
        # This branch is for constants and should be handled appropriately
        # This might need to handle more specific cases or throw an error
        return expr


def extract_sympy_Net(net, sympy_vars=None):
    input_size = net.layers[0].in_features
    x_dreal = [dreal.Variable(f"x{i}") for i in range(1, input_size + 1)]
    V_net_dreal = lyznet.extract_dreal_Net(net, x_dreal)
    V_net_sympy = sympy.sympify(str(V_net_dreal))
    V_x_sympy = [sympy.sympify(str(V_net_dreal.Differentiate(x))) 
                 for x in x_dreal]
    return V_net_sympy, V_x_sympy


def matrix_multiply_dreal(A, B):
    # Ensure A's columns match B's rows
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication.")
    
    # Initialize the result matrix with zeros (dreal expressions)
    result_dreal = [[dreal.Expression(0) for _ in range(len(B[0]))] 
                    for _ in range(len(A))]
    
    # Perform the multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                result_dreal[i][j] += A[i][k] * B[k][j]
    
    return result_dreal


def generate_NN_perturbation_dReal_to_Sympy(W_np, beta_np, symbolic_vars):
    dreal_vars = [dreal.Variable(str(var)) for var in symbolic_vars]
    Wx_dreal = numpy.dot(W_np, dreal_vars) 
    # print("Wx_dreal: ", Wx_dreal)  
    tanh_Wx_dreal = [dreal.tanh(Wx_dreal[j]) for j in range(len(Wx_dreal))]    
    beta_tanh_Wx_dreal = numpy.dot(beta_np, tanh_Wx_dreal)   
    # print("beta_tanh_Wx_dreal: ", beta_tanh_Wx_dreal)
    result_sympy = sympy.Matrix([sympy.sympify(str(entry), evaluate=False) 
                                 for entry in beta_tanh_Wx_dreal])
    return result_sympy


def sympy_matrix_multiply_dreal(A, B, symbolic_vars):
    # Multiply two sympy matrices using dreal multiplication
    # Check if the matrices can be multiplied
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions do not match for multiplication.")
    
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))

    # Convert matrices to dreal
    A_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in A.tolist()]
    B_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in B.tolist()]

    # Perform matrix multiplication
    result_rows = A.shape[0]
    result_cols = B.shape[1]
    result_dreal = [[sum(A_dreal[i][k] * B_dreal[k][j] 
                     for k in range(A.shape[1]))
                     for j in range(result_cols)] for i in range(result_rows)]

    # Convert the result back to sympy
    result_sympy = sympy.Matrix([[sympy.sympify(str(entry), evaluate=False) 
                                 for entry in row] for row in result_dreal])
    return result_sympy


def compute_Dg_zero(system):
    symbolic_vars = system.symbolic_vars
    g_matrix = (system.symbolic_g).T
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))
    g_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in g_matrix.tolist()]
    jacobian_dreal = [[[element.Differentiate(var) for var in dreal_vars] 
                      for element in row] for row in g_dreal]
    
    substitution_dict = {var: 0 for var in dreal_vars}
    evaluated_jacobian = [[[element.Substitute(substitution_dict).Evaluate() 
                          for element in row] for row in layer] 
                          for layer in jacobian_dreal]
    evaluated_jacobian = numpy.array(evaluated_jacobian)
    return evaluated_jacobian


def compute_controller_gain_ELM_loss_numpy(weights, bias, R_inv, 
                                           gT_zero, Dg_zero):
    d = weights.shape[1]
    m = weights.shape[0]
    tanh_b = numpy.tanh(bias)
    # need to be changed for alternative activation functions
    sigma_prime_b = 1 - tanh_b**2  # shape (m, 1)
    sigma_prime_prime_b = -2 * tanh_b * sigma_prime_b  # shape (m, 1)

    grad_V = (weights * sigma_prime_b).T
    # print("grad_V_zero:", grad_V.shape)
    # print("Dg_zero:", Dg_zero.shape)
    # print("gT_zero:", gT_zero.shape)

    hessian_V_tensor = numpy.zeros((d, d, m))

    for i in range(m):
        W_i = weights[i, :].reshape(d, 1)  # Reshape W_i to a column vector
        hessian_V_tensor[:, :, i] = W_i @ W_i.T * sigma_prime_prime_b[i]

    # print("hessian_V_tensor: ", hessian_V_tensor.shape)

    d_Dg_DV = (numpy.einsum('ij,jkl->ikl', gT_zero, hessian_V_tensor) 
               + numpy.einsum('ijk,kl->ijl', Dg_zero, grad_V))
    # print("d_DgT_DVT: ", d_Dg_DV.shape)

    pre_K = - 0.5 * numpy.einsum('ij, jkl', R_inv, d_Dg_DV) 
    # print("pre_K: ", pre_K.shape)
    return pre_K


def compute_controller_gain_ELM_loss_dreal(weights, bias, system):
    symbolic_vars = system.symbolic_vars
    g_matrix = (system.symbolic_g).T
    R_inv_sym = sympy.Matrix(system.R).inv()

    # get u_expr (without beta) using dreal
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))

    sigma_prime_diagonal = [(1 - dreal.tanh(sum(weights[i][j] * dreal_vars[j] 
                            for j in range(len(weights[0]))) + bias[i])**2) 
                            for i in range(len(weights))]

    diagonal_matrix = [[sigma_prime_diagonal[i] if i == j else 0 
                       for j in range(len(weights))] 
                       for i in range(len(weights))]

    R_inv = [[sympy_to_dreal(entry, subs) for entry in row] 
             for row in R_inv_sym.tolist()]
    gT = [[sympy_to_dreal(entry, subs) for entry in row] 
          for row in g_matrix.tolist()]
    WT = [[weights[j][i] for j in range(len(weights))] 
          for i in range(len(weights[0]))]

    gT_WT = matrix_multiply_dreal(gT, WT)

    diagonal_scaled_gT_WT = matrix_multiply_dreal(gT_WT, diagonal_matrix)

    result = matrix_multiply_dreal(R_inv, diagonal_scaled_gT_WT)

    result = [[-0.5 * element for element in row] for row in result]

    jacobian_dreal = [[[element.Differentiate(var) for var in dreal_vars] 
                      for element in row] for row in result]
    
    substitution_dict = {var: 0 for var in dreal_vars}
    evaluated_jacobian = [[[element.Substitute(substitution_dict).Evaluate() 
                          for element in row] for row in layer] 
                          for layer in jacobian_dreal]
    evaluated_jacobian = numpy.array(evaluated_jacobian)
    # the result should be of dimension k x d x m
    transposed_jacobian = numpy.transpose(evaluated_jacobian, (0, 2, 1))
    return transposed_jacobian


def compute_controller_gain_from_ELM_dreal(weights, bias, beta, system):
    symbolic_vars = system.symbolic_vars
    g_matrix = (system.symbolic_g).T
    R_inv_sym = sympy.Matrix(system.R).inv()

    # get u_expr using dreal
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))

    z = numpy.dot(dreal_vars, weights.T) + bias.squeeze(-1)
    h = []
    for j in range(len(weights)):
        h.append(dreal.tanh(z[j]))
    V_dreal = numpy.dot(h, beta.T)
    # print("V = ", V_dreal)

    V_x_dreal = [V_dreal.Differentiate(x) for x in dreal_vars]

    # Convert sympy matrices to dreal
    R_inv_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
                   for row in R_inv_sym.tolist()]
    g_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in g_matrix.tolist()]

    # Perform matrix multiplication in dreal
    u_expr_dreal = []
    for i in range(len(R_inv_dreal)):
        sum_expr = 0
        for j in range(len(g_dreal)):
            temp_sum = 0
            for k in range(len(V_x_dreal)):
                temp_sum += g_dreal[j][k] * V_x_dreal[k]
            sum_expr += R_inv_dreal[i][j] * temp_sum
        u_expr_dreal.append(-1/2 * sum_expr)

    # Compute Jacobian of u_expr at zero to get gain matrix K
    jacobian_dreal = [[func.Differentiate(var) for var in dreal_vars] 
                      for func in u_expr_dreal]
    substitution_dict = {var: 0 for var in dreal_vars}
    evaluated_jacobian = [[element.Substitute(substitution_dict).Evaluate() 
                          for element in row] for row in jacobian_dreal]
    
    K = numpy.array(evaluated_jacobian)
    return K


def get_controller_from_ELM_dreal(weights, bias, beta, R_inv_sym, g_matrix, 
                                  symbolic_vars):
    sys.setrecursionlimit(9999)
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))

    z = numpy.dot(dreal_vars, weights.T) + bias.squeeze(-1)
    h = []
    for j in range(len(weights)):
        h.append(dreal.tanh(z[j]))
    V_dreal = numpy.dot(h, beta.T)
    # print("V = ", V_dreal)

    V_x_dreal = [V_dreal.Differentiate(x) for x in dreal_vars]
    # Convert sympy matrices to dreal
    R_inv_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
                   for row in R_inv_sym.tolist()]
    g_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in g_matrix.tolist()]
    # Perform matrix multiplication in dreal
    u_expr_dreal = []
    for i in range(len(R_inv_dreal)):
        sum_expr = 0
        for j in range(len(g_dreal)):
            temp_sum = 0
            for k in range(len(V_x_dreal)):
                temp_sum += g_dreal[j][k] * V_x_dreal[k]
            sum_expr += R_inv_dreal[i][j] * temp_sum
        u_expr_dreal.append(-1/2 * sum_expr)

    # Convert the result back to sympy; this step takes the most time
    u_matrix_sympy = sympy.Matrix([sympy.sympify(str(expr), evaluate=False) 
                                   for expr in u_expr_dreal])
    return u_matrix_sympy


def get_controller_from_Net_dreal(net, R_inv_sym, g_matrix, symbolic_vars):
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    subs = dict(zip(symbolic_vars, dreal_vars))

    V_net_dreal = lyznet.extract_dreal_Net(net, dreal_vars)
    V_x_dreal = [V_net_dreal.Differentiate(x) for x in dreal_vars]

    # Convert sympy matrices to dreal
    R_inv_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
                   for row in R_inv_sym.tolist()]
    g_dreal = [[sympy_to_dreal(entry, subs) for entry in row] 
               for row in g_matrix.tolist()]

    # Perform matrix multiplication in dreal
    result_dreal = []
    for i in range(len(R_inv_dreal)):
        sum_expr = 0
        for j in range(len(g_dreal)):
            temp_sum = 0
            for k in range(len(V_x_dreal)):
                temp_sum += g_dreal[j][k] * V_x_dreal[k]
            sum_expr += R_inv_dreal[i][j] * temp_sum
        result_dreal.append(-1/2 * sum_expr)

    # Convert the result back to sympy; this step takes the most time
    u_matrix_sympy = sympy.Matrix([sympy.sympify(str(expr), evaluate=False) 
                                  for expr in result_dreal])
    return u_matrix_sympy


def read_controller_from_Net_dreal(net, symbolic_vars):
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    u_net_dreal = lyznet.extract_dreal_UNet(net, dreal_vars)
    u_matrix_sympy = [sympy.sympify(str(expr), evaluate=False) 
                      for expr in u_net_dreal]
    return u_matrix_sympy


def dreal_to_z3(dreal_expr, z3_variables):
    dreal_expr_str = str(dreal_expr)
    var_map = {f"x{i}": var for i, var in enumerate(z3_variables, start=1)}
    
    for dreal_name, z3_var in var_map.items():
        dreal_expr_str = dreal_expr_str.replace(dreal_name, f"var_map['{dreal_name}']")
    
    z3_expr = eval(dreal_expr_str, {"var_map": var_map, "z3": z3})    
    return z3_expr


def compute_jacobian_np_dreal(symbolic_f, symbolic_vars):
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    dreal_f = [sympy_to_dreal(f, dict(zip(symbolic_vars, dreal_vars))) 
               for f in symbolic_f]
    jacobian_dreal = [[func.Differentiate(var) for var in dreal_vars] 
                      for func in dreal_f]
    substitution_dict = {var: 0 for var in dreal_vars}
    evaluated_jacobian = [[element.Substitute(substitution_dict).Evaluate() 
                          for element in row] for row in jacobian_dreal]
    return numpy.array(evaluated_jacobian)


def evaluate_at_origin_dreal(u_expr, symbolic_vars):
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    dreal_u_expr = [sympy_to_dreal(f, dict(zip(symbolic_vars, dreal_vars))) 
                    for f in u_expr]
    substitution_dict = {var: 0 for var in dreal_vars}
    evaluated_u_expr = [element.Substitute(substitution_dict).Evaluate() 
                        for element in dreal_u_expr]
    return numpy.array(evaluated_u_expr)


def compute_jacobian_sp_dreal(symbolic_f, symbolic_vars):
    dreal_vars = [dreal.Variable(str(v)) for v in symbolic_vars]
    dreal_f = [sympy_to_dreal(f, dict(zip(symbolic_vars, dreal_vars))) 
               for f in symbolic_f]
    jacobian_dreal = [[func.Differentiate(var) for var in dreal_vars] 
                      for func in dreal_f]
    jacobian_sympy = sympy.Matrix([[sympy.sympify(str(entry)) for entry in row] 
                                   for row in jacobian_dreal])
    return jacobian_sympy


def get_x_bound(x, xlim): 
    bounds_conditions = []
    for i in range(len(xlim)):
        lower_bound = x[i] >= xlim[i][0]
        upper_bound = x[i] <= xlim[i][1]
        bounds_conditions.append(dreal.logical_and(lower_bound, upper_bound))
    all_bounds = dreal.logical_and(*bounds_conditions)
    return all_bounds


def get_bound(x, xlim, V, c1_V=0.0, c2_V=1.0): 
    bounds_conditions = []
    for i in range(len(xlim)):
        lower_bound = x[i] >= xlim[i][0]
        upper_bound = x[i] <= xlim[i][1]
        bounds_conditions.append(dreal.logical_and(lower_bound, upper_bound))
    all_bounds = dreal.logical_and(*bounds_conditions)
    c1_V = dreal.Expression(c1_V)
    c2_V = dreal.Expression(c2_V)
    vars_in_bound = dreal.logical_and(c1_V <= V, V <= c2_V)
    x_bound = dreal.logical_and(all_bounds, vars_in_bound)
    return x_bound


def config_dReal(number_of_jobs=None, tol=None):
    config = dreal.Config()
    config.use_polytope_in_forall = True
    config.use_local_optimization = True
    config.precision = tol
    config.number_of_jobs = number_of_jobs
    return config


def test_volume(vdp_system, net, c2, data):
    x_data, y_data = data[:, :-1], data[:, -1]
    outputs = net(torch.Tensor(x_data)).squeeze().detach().numpy()
    print('_' * 50)
    print("Testing volume of sublevel set of neural Lyapunov function...")
    print(f"The size of data is {len(outputs)}")

    num_cases = numpy.count_nonzero(y_data < 1)
    print(f"The size of ROA (approximated by data) is {num_cases}")
    count = 0

    for i in range(len(outputs)):
        if outputs[i] <= c2:
            count += 1

    ratio = count / num_cases * 100
    print(f"The approximate volume ratio of verified ROA is {ratio:.2f}%")

    return ratio


def test_volume_elm(system, weights, bias, beta, c2_V, test_data):
    x_data, y_data = test_data[:, :-1], test_data[:, -1]
    outputs = numpy.array(
        [numpy.dot(numpy.tanh(numpy.dot(x_point, weights.T) 
         + bias.squeeze(-1)), beta) for x_point in x_data]
        ).squeeze()
    print('_' * 50)
    print("Testing volume of sublevel set of neural Lyapunov function...")
    print(f"The size of data is {len(outputs)}")

    num_cases = numpy.count_nonzero(y_data < 1)
    print(f"The size of ROA (approximated by data) is {num_cases}")
    count = 0

    for output in outputs:
        if output <= c2_V:
            count += 1

    ratio = count / num_cases * 100 if num_cases > 0 else 0
    print(f"The approximate volume ratio of verified ROA is {ratio:.2f}%")

    return ratio


def test_volume_sos(vdp_system, func, c2, data):
    x_data, y_data = data[:, :-1], data[:, -1]
    x1, x2 = x_data[:, 0], x_data[:, 1]
    outputs = func(x1, x2)
    print('_' * 50)
    print("Testing volume of sublevel set of SOS Lyapunov function...")
    print(f"The size of data is {len(outputs)}")

    num_cases = numpy.count_nonzero(y_data < 1)
    print(f"The size of ROA (approximated by data) is {num_cases}")
    count = 0

    for i in range(len(outputs)):
        if outputs[i] <= c2:
            count += 1

    ratio = count / num_cases * 100
    print(f"The approximate volume ratio of verified ROA is {ratio:.2f}%")

    return ratio
