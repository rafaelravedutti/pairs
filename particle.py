import part_prot as pt

dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

mass = pt.add_property('mass', 1.0)
position = pt.add_vector_property('position')
velocity = pt.add_vector_property('velocity')
force = pt.add_vector_property('force')

grid_config = [[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]]
pt.setup_grid(grid_config)
pt.create_particle_lattice(grid_config, spacing=[1.0, 1.0, 1.0])
pt.setup_cell_lists(cutoff_radius + skin)
pt.set_timesteps(100)

force[i].set(0.0)
for i, j, delta, rsq in pt.particle_pairs(cutoff_radius, position):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    force[i].add(delta * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon)

for i in pt.particles():
    velocity[i].add(dt * force[i] / mass[i])
    position[i].add(dt * velocity[i])
