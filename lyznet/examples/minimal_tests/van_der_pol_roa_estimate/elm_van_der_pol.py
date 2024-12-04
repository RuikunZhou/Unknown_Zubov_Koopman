import sympy 
import lyznet

lyznet.utils.set_random_seed(seed=123)

# Define dynamics
mu = 1.0
x1, x2 = sympy.symbols('x1 x2')
f_vdp = [-x2, x1 - mu * (1 - x1**2) * x2]
domain_vdp = [[-2.5, 2.5], [-3.5, 3.5]]
sys_name = f"elm_van_der_pol_mu_{mu}.py"
vdp_system = lyznet.DynamicalSystem(f_vdp, domain_vdp, sys_name)

print("System dynamics: x' = ", vdp_system.symbolic_f)
print("Domain: ", vdp_system.domain)

c1_P = lyznet.local_stability_verifier(vdp_system)
c2_P = lyznet.quadratic_reach_verifier(vdp_system, c1_P)

W, b, beta, model_path = lyznet.numpy_elm_learner(
    vdp_system, num_hidden_units=100, num_colloc_pts=20000, 
    lambda_reg=0.1, 
    loss_mode="Zubov"
    )

c1_V, c2_V = lyznet.numpy_elm_verifier(vdp_system, W, b, beta, c2_P)

lyznet.plot_V(vdp_system, elm_model=[W, b, beta], model_path=model_path, 
              phase_portrait=True, c2_V=c2_V, c2_P=c2_P, c1_V=c1_V)
