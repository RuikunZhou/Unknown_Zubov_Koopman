import sympy as sp
import lyznet

lyznet.utils.set_random_seed(seed=123)

mu = 1.0
x1, x2 = sp.symbols('x1 x2')
f = sp.Matrix([x2, -x1 + mu * (1 - x1**2) * x2])
g = sp.Matrix([0, 1.0])

domain = [[-2.0, 2.0]] * 2

sys_name = f"van_der_pol_mu_{mu}_domain{domain}"
system = lyznet.ControlAffineSystem(f, g, domain, sys_name)

initial_u = sp.Matrix([-mu * (1 - x1**2) * x2 - 10*x2])

lyznet.elm_pi(system, num_of_iters=10, width=800, 
              initial_u=initial_u, 
              plot_each_iteration=True, 
              num_colloc_pts=3000, final_plot=True, final_test=True
              )

# lyznet.neural_pi(system, 
#                  num_of_iters=10, lr=0.001, layer=2, width=20, 
#                  num_colloc_pts=300000, max_epoch=5,
#                  plot_each_iteration=True, verify=True)
