import pairs
import sys


def lj(i, j):
    sr2 = 1.0 / squared_distance(i, j)
    sr6 = sr2 * sr2 * sr2 * sigma6[i, j]
    apply(force, delta(i, j) * (48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[i, j]))

def euler(i):
    linear_velocity[i] += dt * force[i] / mass[i]
    position[i] += dt * linear_velocity[i]


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

psim = pairs.simulation("lj", [pairs.point_mass()], timesteps=200, double_prec=True, debug=True)
psim.add_position('position')
psim.add_property('mass', pairs.real(), 1.0)
psim.add_property('linear_velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'epsilon', pairs.real(), [sigma for i in range(ntypes * ntypes)])
psim.add_feature_property('type', 'sigma6', pairs.real(), [epsilon for i in range(ntypes * ntypes)])
psim.set_domain([0.0, 0.0, 0.0, 53.747078, 53.747078, 53.747078])
psim.read_particle_data("data/minimd_setup_32x32x32.input", ['type', 'mass', 'position', 'linear_velocity', 'flags'], pairs.point_mass())
psim.reneighbor_every(20)
#psim.compute_half()
psim.build_neighbor_lists(cutoff_radius + skin)
psim.vtk_output(f"output/lj_{target}")
psim.compute(lj, cutoff_radius)
psim.compute(euler, symbols={'dt': dt})

if target == 'gpu':
    psim.target(pairs.target_gpu())
else:
    psim.target(pairs.target_cpu())

psim.generate()
