import pairs


dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

psim = pairs.simulation("lj")
mass = psim.add_real_property('mass', 1.0)
position = psim.add_vector_property('position')
velocity = psim.add_vector_property('velocity')
force = psim.add_vector_property('force', vol=True)
psim.from_file("data/minimd_setup_4x4x4.input", ['mass', 'position', 'velocity'])
psim.create_cell_lists(2.8, 2.8)
psim.create_neighbor_lists()
psim.periodic(2.8)
psim.vtk_output("output/test")

for i, j, delta, rsq in psim.particle_pairs(cutoff_radius, position):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    f = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon
    force[i].add(delta * f)

for i in psim.particles():
    velocity[i].add(dt * force[i] / mass[i])
    position[i].add(dt * velocity[i])

psim.generate()
