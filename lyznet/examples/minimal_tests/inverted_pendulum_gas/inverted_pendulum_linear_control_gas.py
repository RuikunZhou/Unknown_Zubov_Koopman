# This script demonstrates the use of LyzNet for verifying global stability 
# of an inverted pendulum under linear control

import sympy 
import lyznet

lyznet.utils.set_random_seed(47)

x1, x2 = sympy.symbols('x1 x2')
k = [4.4142, 2.3163]
f = [x2, sympy.sin(x1) - x2 - (k[0]*x1 + k[1]*x2)]
domain = [[-16.0, 16.0]] * 2
sys_name = "ex5_pendulum_linear_control"
system = lyznet.DynamicalSystem(f, domain, sys_name)

print("System dynamics: x' = ", system.symbolic_f)
c1_P = lyznet.local_stability_verifier(system)
