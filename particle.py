import part_prot as pt
from ast.layouts import Layout_SoA

dt = 0.005
cutoff_radius = 2.5
skin = 0.3
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6

psim = pt.simulation()
mass = psim.add_real_property('mass', 1.0)
position = psim.add_vector_property('position', layout=Layout_SoA)
velocity = psim.add_vector_property('velocity')
force = psim.add_vector_property('force', vol=True)

grid_config = [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]
psim.setup_grid(grid_config)
psim.create_particle_lattice(grid_config, spacing=[1.0, 1.0, 1.0])

for i, j, delta, rsq in psim.particle_pairs(cutoff_radius, position):
    sr2 = 1.0 / rsq
    sr6 = sr2 * sr2 * sr2 * sigma6
    f = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon
    force[i].add(delta * f)

for i in psim.particles():
    velocity[i].add(dt * force[i] / mass[i])
    position[i].add(dt * velocity[i])

psim.generate()
