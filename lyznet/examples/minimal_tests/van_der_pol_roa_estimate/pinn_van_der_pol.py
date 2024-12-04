import sympy 
import lyznet

lyznet.utils.set_random_seed()

# Define dynamics
mu = 1.0
x1, x2 = sympy.symbols('x1 x2')
f_vdp = [-x2, x1 - mu * (1 - x1**2) * x2]
domain_vdp = [[-2.5, 2.5], [-3.5, 3.5]]
sys_name = f"pinn_van_der_pol_mu_{mu}.py"
vdp_system = lyznet.DynamicalSystem(f_vdp, domain_vdp, sys_name)

print("System dynamics: x' = ", vdp_system.symbolic_f)
print("Domain: ", vdp_system.domain)

data = lyznet.generate_data(vdp_system, n_samples=3000)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(vdp_system)
# Call the quadratic verifier
c2_P = lyznet.quadratic_reach_verifier(vdp_system, c1_P)


# # Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(vdp_system, lr=0.001, data=data, 
                                        layer=2, width=10, 
                                        num_colloc_pts=300000, max_epoch=5,
                                        loss_mode="Zubov")

# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(vdp_system, net, c2_P)

lyznet.plot_V(vdp_system, net, model_path, c2_V=c2_V, c2_P=c2_P,
              phase_portrait=True)
