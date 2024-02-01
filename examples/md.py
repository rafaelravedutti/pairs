import pairs
import sys


def lennard_jones(i, j):
    sr2 = 1.0 / squared_distance(i, j)
    sr6 = sr2 * sr2 * sr2 * sigma6[i, j]
    apply(force, delta(i, j) * (48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[i, j]))


def initial_integrate(i):
    linear_velocity[i] += (dt * 0.5) * force[i] / mass[i]
    position[i] += dt * linear_velocity[i]


def final_integrate(i):
    linear_velocity[i] += (dt * 0.5) * force[i] / mass[i]


cmd = sys.argv[0]
target = sys.argv[1] if len(sys.argv[1]) > 1 else "none"
if target != 'cpu' and target != 'gpu':
    print(f"Invalid target, use {cmd} <cpu/gpu>")

dt = 0.005
cutoff_radius = 2.5
skin = 0.3
ntypes = 4
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6
nx = 32
ny = 32
nz = 32
rho = 0.8442
temp = 1.44

psim = pairs.simulation("md", [pairs.point_mass()], timesteps=200, double_prec=True)

if target == 'gpu':
    psim.target(pairs.target_gpu())
else:
    psim.target(pairs.target_cpu())

psim.add_position('position')
psim.add_property('mass', pairs.real(), 1.0)
psim.add_property('linear_velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'epsilon', pairs.real(), [sigma for i in range(ntypes * ntypes)])
psim.add_feature_property('type', 'sigma6', pairs.real(), [epsilon for i in range(ntypes * ntypes)])

psim.copper_fcc_lattice(nx, ny, nz, rho, temp, ntypes)
psim.set_domain_partitioner(pairs.block_forest())
#psim.set_domain_partitioner(pairs.regular_domain_partitioner())
psim.compute_thermo(100)

psim.reneighbor_every(20)
#psim.compute_half()
psim.build_neighbor_lists(cutoff_radius + skin)
#psim.vtk_output(f"output/md_{target}")

psim.compute(initial_integrate, symbols={'dt': dt}, pre_step=True, skip_first=True)
psim.compute(lennard_jones, cutoff_radius)
psim.compute(final_integrate, symbols={'dt': dt}, skip_first=True)
psim.generate()
