import pairs


def lj(i, j):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    force[i] += delta * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon


def euler(i):
    velocity[i] += dt * force[i] / mass[i]
    position[i] += dt * velocity[i]


dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

psim = pairs.simulation("lj_ns", debug=True)
psim.add_real_property('mass', 1.0)
psim.add_position('position')
psim.add_vector_property('velocity')
psim.add_vector_property('force', vol=True)
psim.from_file("data/minimd_setup_4x4x4.input", ['mass', 'position', 'velocity'])
psim.create_cell_lists(2.8, 2.8)
psim.create_neighbor_lists()
psim.periodic(2.8)
psim.vtk_output("output/test")
psim.compute(lj, cutoff_radius, {'sigma6': sigma6, 'epsilon': epsilon})
psim.compute(euler, symbols={'dt': dt})
psim.target(pairs.target_gpu())
psim.generate()
