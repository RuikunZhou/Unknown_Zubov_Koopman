import sympy 
import lyznet

lyznet.utils.set_random_seed()

# Define dynamics
x1, x2, x3 = sympy.symbols('x1 x2 x3')
f_3d = [x2 + 2*x2*x3, x3, -0.5*x1 - 2 * x2 - x3]
domain_3d = [[-2., 2.], [-2., 2.], [-2., 2.]]
sys_name = f"3d_polynomial.py"
poly3d_system = lyznet.DynamicalSystem(f_3d, domain_3d, sys_name)

print("System dynamics: x' = ", poly3d_system.symbolic_f)
print("Domain: ", poly3d_system.domain)

# Call the local stability verifier
c1_P = lyznet.local_stability_verifier(poly3d_system )
# Call the quadratic verifier
c2_P = lyznet.quadratic_reach_verifier(poly3d_system , c1_P)

# # Generate data (needed for data-augmented learner)
data = lyznet.generate_data(poly3d_system, n_samples=3000)

# # Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(poly3d_system, data=data, lr=0.001, 
                                        layer=2, width=10, 
                                        num_colloc_pts=300, max_epoch=2,
                                        loss_mode="Zubov")

# Call the neural lyapunov verifier
c1_V, c2_V = lyznet.neural_verifier(poly3d_system , net, c2_P)

# Compare verified ROA with SOS
def sos_V(x1, x2, x3):
    return (-6.10426074423e-06*x1-4.06777321759e-05*x2-3.57328558801e-06*x3+0.0684671222286*x2*x3 \
            +0.0199530135501*x1**2+0.109233890158*x1*x2+0.00968425880286*x1*x3+0.226917667776*x2**2+ \
            0.107884672479*x3**2+0.141211358663*x1*x2*x3+0.117852442686*x2**2*x3-0.0193604864816*x2*x3**2 \
            +0.0469070864902*x1**3+0.146559761348*x1**2*x2+0.239345899049*x1*x2**2-0.0788563641221*x2**3+ \
            0.0496173659587*x1**2*x3+0.148149681202*x1*x3**2+0.0737684860329*x3**3+0.0430416386066*x1**4+ \
            0.0554138191343*x1**3*x2+0.104314242035*x1**2*x2**2+0.0753851579545*x1*x2**3-0.031642849054*x2**4 
            +0.0872692593421*x1**3*x3+0.0363213995716*x1**2*x2*x3+0.0536971088944*x1*x2**2*x3+0.0530436043357*x2**3*x3 \
            +0.094189563815*x1**2*x3**2+0.0656210422054*x1*x2*x3**2+0.129830365772*x2**2*x3**2-0.0194390215195*x1*x3**3+ \
            0.0395537533818*x2*x3**3+0.0702667807527*x3**4+0.00373059727555*x1**5-0.0027543586002*x1**4*x2-0.112532223488*x1**3*x2**2- \
            0.114173075795*x1**2*x2**3-0.0996390499927*x1*x2**4+0.0032177862745*x2**5+0.0274465226226*x1**4*x3-0.0296803863731*x1**3*x2*x3 \
            -0.237410775375*x1**2*x2**2*x3-0.0687264583146*x1*x2**3*x3-0.0325861037593*x2**4*x3+0.0498018353315*x1**3*x3**2
            -0.0885409471311*x1**2*x2*x3**2-0.0433342522465*x1*x2**2*x3**2+0.0873574339842*x2**3*x3**2-0.0275699253588*x1**2*x3**3-
            0.0636363683216*x1*x2*x3**3+0.0222643739555*x2**2*x3**3-0.0129139756476*x1*x3**4+0.0217352508536*x2*x3**4-0.0103881340625*x3**5 \
            +0.0010555280061*x1**6+0.002941548768*x1**5*x2+0.0061096500756*x1**4*x2**2+0.00275466771368*x1**3*x2**3+0.0361154084211*x1**2*x2**4 \
            +0.0240451671358*x1*x2**5+0.0508608042863*x2**6+0.00523062492754*x1**5*x3+0.0172464159095*x1**4*x2*x3+0.0143685513779*x1**3*x2**2*x3 \
            +0.0196926077681*x1**2*x2**3*x3+0.0458934789286*x1*x2**4*x3+0.0171572306286*x2**5*x3+0.0149035311608*x1**4*x3**2+0.0384477076932*x1**3*x2*x3**2 \
            -0.00699487355219*x1**2*x2**2*x3**2-0.0170879007215*x1*x2**3*x3**2+0.0139302917135*x2**4*x3**2+0.0268588119488*x1**3*x3**3+ \
            0.0331836580029*x1**2*x2*x3**3-0.0327738651784*x1*x2**2*x3**3+0.00238887448885*x2**3*x3**3+0.0259147699207*x1**2*x3**4 \
            -0.0212207091782*x1*x2*x3**4+0.000254083480502*x2**2*x3**4-0.0025292561244*x1*x3**5-0.012893805377*x2*x3**5+0.00703902624577*x3**6)


sos_V_sympy = sos_V(x1, x2)
c1_SOS, c2_SOS = lyznet.sos_reach_verifier(poly3d_system, sos_V_sympy, c2_P)

lyznet.plot_V(poly3d_system, net, model_path, 
              V_list=[sos_V], c_lists=[[c2_SOS]], c2_V=c2_V, c2_P=c2_P,
              phase_portrait=True)

# test_data = lyznet.generate_data(poly3d_system, n_samples=90000)
# volume_percent = lyznet.utils.test_volume(poly3d_system, net, c2_V, test_data)
# sos_volume_percent = lyznet.utils.test_volume_sos(poly3d_system, sos_V, c2_SOS, 
#                                                   test_data)
