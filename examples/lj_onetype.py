import pairs
import sys


def lj(i, j):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    force[i] += delta * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon


def euler(i):
    velocity[i] += dt * force[i] / mass[i]
    position[i] += dt * velocity[i]


cmd = sys.argv[0]
target = sys.argv[1] if len(sys.argv[1]) > 1 else "none"
if target != 'cpu' and target != 'gpu':
    print(f"Invalid target, use {cmd} <cpu/gpu>")


dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

psim = pairs.simulation("lj", debug=True)
psim.add_real_property('mass', 1.0)
psim.add_position('position')
psim.add_vector_property('velocity')
psim.add_vector_property('force', vol=True)
psim.from_file("data/minimd_setup_32x32x32.input", ['mass', 'position', 'velocity'])
psim.build_neighbor_lists(cutoff_radius + skin)
psim.vtk_output(f"output/test_{target}")
psim.compute(lj, cutoff_radius, {'sigma6': sigma6, 'epsilon': epsilon})
psim.compute(euler, symbols={'dt': dt})

if target == 'gpu':
    psim.target(pairs.target_gpu())
else:
    psim.target(pairs.target_cpu())

psim.generate()