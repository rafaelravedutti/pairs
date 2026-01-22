import math
import pairs

def update_mass_and_inertia(i):
    rotation_matrix[i] = diagonal_matrix(1.0)
    rotation[i] = default_quaternion()

    if is_sphere(i):
        inv_inertia[i] = inversed(diagonal_matrix(0.4 * mass[i] * radius[i] * radius[i]))

    else:
        mass[i] = infinity
        inv_inertia[i] = 0.0

def spring_dashpot(i, j):
    delta_ij = -penetration_depth(i, j)
    skip_when(delta_ij < 0.0)     
    
    velocity_wf_i = linear_velocity[i] + cross(angular_velocity[i], contact_point(i, j) - position[i])
    velocity_wf_j = linear_velocity[j] + cross(angular_velocity[j], contact_point(i, j) - position[j])
    
    rel_vel = -(velocity_wf_i - velocity_wf_j)
    rel_vel_n = dot(rel_vel, contact_normal(i, j))
    rel_vel_t = rel_vel - rel_vel_n * contact_normal(i, j)

    fNabs = stiffness[i,j] * delta_ij + damping_norm[i,j] * rel_vel_n
    fN = fNabs * contact_normal(i, j)

    fTabs = min(damping_tan[i,j] * length(rel_vel_t), friction[i, j] * fNabs)
    fT = fTabs * normalized(rel_vel_t)

    partial_force = fN + fT
    apply(force, partial_force)
    apply(torque, cross(contact_point(i, j) - position, partial_force))

def euler(i):
    skip_when(is_fixed(i) or is_infinite(i))
    inv_mass = 1.0 / mass[i]
    position[i] +=  0.5 * inv_mass * force[i] * dt * dt + linear_velocity[i] * dt
    linear_velocity[i] += inv_mass * force[i] * dt
    wdot = rotation_matrix[i] * (inv_inertia[i] * torque[i]) * transposed(rotation_matrix[i])
    phi = angular_velocity[i] * dt + 0.5 * wdot * dt * dt
    rotation[i] = quaternion(phi, length(phi)) * rotation[i]
    rotation_matrix[i] = quaternion_to_rotation_matrix(rotation[i])
    angular_velocity[i] += wdot * dt


psim = pairs.Simulation(particle_capacity=10000000)

# Add position property
psim.add_position('position')

# Add shapes and define their geometric properties (required internally for contact detection)
psim.add_shape(pairs.sphere('radius'))

# Add properties
#-------------------------------------------------------------------------------------------------
psim.add_property('radius',         pairs.real(),       pairs.on_reneighbor()) # Required by spheres as defined above
psim.add_property('mass',           pairs.real(),       pairs.on_reneighbor())
psim.add_property('inv_inertia',    pairs.matrix(),     pairs.on_reneighbor())
psim.add_property('rotation_matrix',    pairs.matrix(),     pairs.always())
psim.add_property('rotation',           pairs.quaternion(), pairs.always())
psim.add_property('linear_velocity',    pairs.vector(),     pairs.always())
psim.add_property('angular_velocity',   pairs.vector(),     pairs.always())
psim.add_property('force',              pairs.vector(),     pairs.never())
psim.add_property('torque',             pairs.vector(),     pairs.never())
#-------------------------------------------------------------------------------------------------

ntypes = 1
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'stiffness', pairs.real())
psim.add_feature_property('type', 'damping_norm', pairs.real())
psim.add_feature_property('type', 'damping_tan', pairs.real())
psim.add_feature_property('type', 'friction', pairs.real())

# psim.set_domain_partitioner(pairs.regular_domain_partitioner())
psim.set_domain_partitioner(pairs.block_forest())
psim.pbc([True, True, True])
psim.build_cell_lists(store_neighbors_per_cell=False, use_halo_cells=False, optimize_halo_paddings=False)

psim.compute(update_mass_and_inertia, symbols={'infinity': math.inf })
psim.compute(spring_dashpot, profile=False)
psim.compute(euler, parameters={'dt': pairs.real()})

psim.generate()

