import pairs
from ir.layouts import Layout_SoA

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

#def lj(i, j):
#    sr2 = 1.0 / rsq(i, j)
#    sr6 = sr2 * sr2 * sr2 * sigma6
#    force[i] += delta(i, j) * 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon

#def euler(i):
#    velocity[i] += dt * force[i] / mass[i]
#    position[i] += dt * velocity[i]

# psim.compute(lj)
# psim.compute(euler)

#grid = psim.grid_3d(0.0, 4.0, 0.0, 4.0, 0.0, 4.0)
#psim.create_particle_lattice(grid, spacing=[0.82323, 0.82323, 0.82323])
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
