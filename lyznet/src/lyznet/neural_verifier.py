import time 

import numpy as np
import dreal 

import lyznet


def extract_dreal_Net(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_UNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    U_net = [np.dot(h, final_layer_weight[i]) + final_layer_bias[i] 
             for i in range(final_layer_weight.shape[0])]
    # U_net = np.dot(h, final_layer_weight.T) + final_layer_bias
    return U_net


def extract_dreal_PolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]
    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    h = x
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net


def extract_dreal_HomoNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [dreal.tanh(z[j]) for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    
    input_layer_weight_norm = np.linalg.norm(weights[0])

    return V_net * (norm ** model.deg), input_layer_weight_norm


def extract_dreal_HomoPolyNet(model, x):
    layers = len(model.layers) 
    weights = [layer.weight.data.cpu().numpy() for layer in model.layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.layers]

    final_layer_weight = model.final_layer.weight.data.cpu().numpy()
    final_layer_bias = model.final_layer.bias.data.cpu().numpy()

    norm = dreal.sqrt(sum(xi * xi for xi in x))
    h = [xi / norm for xi in x]
    for i in range(layers):
        z = np.dot(h, weights[i].T) + biases[i]
        h = [z[j]**2 for j in range(len(weights[i]))]
    
    V_net = (np.dot(h, final_layer_weight.T) + final_layer_bias)[0]
    return V_net * (norm ** model.deg)


def extract_dreal_SimpleNet(model, x):
    d = len(model.initial_layers)    
    weights = [layer.weight.data.cpu().numpy() 
               for layer in model.initial_layers]
    biases = [layer.bias.data.cpu().numpy() for layer in model.initial_layers]
    
    h = []
    for i in range(d):
        xi = x[i]  
        z = xi * weights[i][0, 0] + biases[i][0]  
        h_i = dreal.tanh(z) 
        h.append(h_i)
    
    final_output = sum([h_i * h_i for h_i in h])    
    return final_output


def neural_verifier(system, model, c2_P=None, c1_V=0.1, c2_V=1, 
                    tol=1e-4, accuracy=1e-2, 
                    net_type=None, number_of_jobs=32, verifier=None):
    # {x^TPx<=c2_P}: target quadratic-Lyapunov level set 
    # c1_V: target Lyapunov level set if c2_P is not specified
    # c2_V: maximal level to be verified

    config = lyznet.utils.config_dReal(number_of_jobs=number_of_jobs, tol=tol)
    xlim = system.domain

    # Create dReal variables based on the number of symbolic variables
    x = [dreal.Variable(f"x{i}") 
         for i in range(1, len(system.symbolic_vars) + 1)]

    f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, x))
            )
        for expr in system.symbolic_f
        ]

    # print("dReal expressions of f: ", f)

    if net_type == "Simple":
        V_learn = extract_dreal_SimpleNet(model, x)
    elif net_type == "Homo": 
        V_learn, norm_W = extract_dreal_HomoNet(model, x)        
    elif net_type == "Poly":
        V_learn = extract_dreal_PolyNet(model, x)
    elif net_type == "HomoPoly":
        V_learn = extract_dreal_HomoPolyNet(model, x)
    else:
        V_learn = extract_dreal_Net(model, x)
    print("V = ", V_learn.Expand())

    lie_derivative_of_V = dreal.Expression(0)
    for i in range(len(x)):
        lie_derivative_of_V += f[i] * V_learn.Differentiate(x[i])

    # If homogeneous verifier is called, do the following: 
    if verifier == "Homo": 
        # config = lyznet.utils.config_dReal(number_of_jobs=32, tol=1e-7)
        norm = dreal.sqrt(sum(xi * xi for xi in x)) 
        unit_sphere = (norm == 1)
        condition_V = dreal.logical_imply(unit_sphere, V_learn >= 1e-7)
        condition_dV = dreal.logical_imply(
            unit_sphere, lie_derivative_of_V <= -1e-7
            )
        condition = dreal.logical_and(condition_V, condition_dV)
        start_time = time.time()
        result = dreal.CheckSatisfiability(
            dreal.logical_not(condition), config
            )
        if result is None:
            print("Global stability verified for homogeneous vector field!")
            # print(f"The norm of the weight matrix is: {norm_W}")
        else:
            print(result)
            print("Stability cannot be verified for homogeneous vector field!")
        end_time = time.time()
        print(f"Time taken for verifying Lyapunov function of {system.name}: " 
              f"{end_time - start_time} seconds.\n")
        return 1, 1

    quad_V = dreal.Expression(0)
    for i in range(len(x)):
        for j in range(len(x)):
            quad_V += x[i] * system.P[i][j] * x[j]
    
    if c2_P is not None:
        target = quad_V <= c2_P

    start_time = time.time()

    def Check_inclusion(c1):
        x_bound = lyznet.utils.get_bound(x, xlim, V_learn, c2_V=c1)
        condition = dreal.logical_imply(x_bound, target)
        return dreal.CheckSatisfiability(dreal.logical_not(condition), config)
 
    print('_' * 50)
    print("Verifying neural Lyapunov function:")

    if c2_P is not None:
        c1_V = lyznet.utils.bisection_glb(Check_inclusion, 0, 1, accuracy)
        print(f"Verified V<={c1_V} is contained in x^TPx<={c2_P}.")
    else:
        print(f"Target level set not specificed. Set it to be V<={c1_V}.")        
    c2_V = lyznet.reach_verifier_dreal(system, x, V_learn, f, c1_V, c_max=c2_V, 
                                       tol=tol, accuracy=accuracy,
                                       number_of_jobs=number_of_jobs)
    print(f"Verified V<={c2_V} will reach V<={c1_V}.")
    end_time = time.time()
    print(f"Time taken for verifying Lyapunov function of {system.name}: " 
          f"{end_time - start_time} seconds.\n")

    return c1_V, c2_V
