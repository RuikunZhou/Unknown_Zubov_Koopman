import time 
# import z3
import dreal
import lyznet


def z3_global_quadratic_verifier(system, eps=1e-5):
    print('_' * 50)
    print("Verifying global stability using quadratic Lyapunov function "
          "(with Z3): ")

    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])
    # print(lie_derivative_of_V_dreal)

    norm_x_squared = sum([x**2 for x in z3_x])

    solver = z3.Solver()

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
    print("DV*f: ", lie_derivative_of_V_z3)

    solver.add(lie_derivative_of_V_z3 > -eps*norm_x_squared)
    
    result = solver.check()

    if result == z3.unsat:
        print("Verified: The EP is globally asymptotically stable.")
    else:
        print("Cannot verify global asymptotic stability. "
              "Counterexample: ")
        print(solver.model())


def verify_quadratic_level(system, c_max=100, eps=1e-5, accuracy=1e-4):
    z3_x = [z3.Real(f"x{i}") for i in range(1, len(system.symbolic_vars) + 1)]
    norm_x_squared = sum([x**2 for x in z3_x])

    xPx = sum([z3_x[i] * sum([system.P[i][j] * z3_x[j] 
               for j in range(len(z3_x))]) for i in range(len(z3_x))])

    norm_x_squared = sum([x**2 for x in z3_x])

    dreal_x = [
        dreal.Variable(f"x{i}") 
        for i in range(1, len(system.symbolic_vars) + 1)
        ]
    dreal_V = dreal.Expression(0)
    for i in range(len(dreal_x)):
        for j in range(len(dreal_x)):
            dreal_V += dreal_x[i] * system.P[i][j] * dreal_x[j]

    dreal_f = [
        lyznet.utils.sympy_to_dreal(
            expr, dict(zip(system.symbolic_vars, dreal_x))
            )
        for expr in system.symbolic_f
        ]

    lie_derivative_of_V_dreal = dreal.Expression(0)
    for i in range(len(dreal_x)):
        lie_derivative_of_V_dreal += dreal_f[i] * dreal_V.Differentiate(
            dreal_x[i])

    lie_derivative_of_V_z3 = lyznet.utils.dreal_to_z3(
        lie_derivative_of_V_dreal, z3_x)
 
    # print("DV*f: ", lie_derivative_of_V_z3)

    def verify_level_c(c):
        solver = z3.Solver()
        solver.add(z3.And(xPx <= c, 
                          lie_derivative_of_V_z3 + eps * norm_x_squared > 0))

        result = solver.check()
        if result == z3.unsat:
            return None
        else:
            return solver.model()

    lyznet.tik()
    c = lyznet.utils.bisection_glb(verify_level_c, 0, c_max, accuracy=accuracy)
    print(f"Region of attraction verified for x^TPx<={c}.")
    lyznet.tok()
    return c
