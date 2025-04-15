import math
import pairs
import sys
import os
        
def update_mass_and_inertia(i):
    rotation_matrix[i] = diagonal_matrix(1.0)
    rotation[i] = default_quaternion()

    if is_sphere(i):
        inv_inertia[i] = inversed(diagonal_matrix(0.4 * mass[i] * radius[i] * radius[i]))

    elif is_box(i):
        inv_inertia[i] = inversed(diagonal_matrix (
            edge_length[i][1]*edge_length[i][1] + edge_length[i][2]*edge_length[i][2],
            edge_length[i][0]*edge_length[i][0] + edge_length[i][2]*edge_length[i][2],
            edge_length[i][0]*edge_length[i][0] + edge_length[i][1]*edge_length[i][1]) * (mass[i] / 12.0))

    elif is_halfspace(i):
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

    fNabs = stiffness[i,j] * delta_ij + max(damping_norm[i,j] * rel_vel_n, 0.0)
    fN = fNabs * contact_normal(i, j)

    fTabs = min(damping_tan[i,j] * length(rel_vel_t), friction[i, j] * fNabs)
    fT =  fTabs * normalized(rel_vel_t)

    partial_force = fN + fT
    apply(force, partial_force)
    apply(torque, cross(contact_point(i, j) - position[i], partial_force))

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

def gravity(i):
    force[i][2] -= mass[i] * gravity_SI


file_name = os.path.basename(__file__)
file_name_without_extension = os.path.splitext(file_name)[0]

psim = pairs.simulation(
    file_name_without_extension,
    [pairs.sphere(), pairs.halfspace(), pairs.box()],
    double_prec=True,
    particle_capacity=1000000,
    neighbor_capacity=20,
    debug=True)


target = sys.argv[1] if len(sys.argv[1]) > 1 else "none"

if target == 'gpu':
    psim.target(pairs.target_gpu())
elif target == 'cpu':
    psim.target(pairs.target_cpu())
else:
    print(f"Invalid target, use {sys.argv[0]} <cpu/gpu>")

psim.add_position('position')
psim.add_property('mass', pairs.real())
psim.add_property('linear_velocity', pairs.vector())
psim.add_property('angular_velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)
psim.add_property('torque', pairs.vector(), volatile=True)
psim.add_property('radius', pairs.real())
psim.add_property('normal', pairs.vector())
psim.add_property('inv_inertia', pairs.matrix())
psim.add_property('rotation_matrix', pairs.matrix())
psim.add_property('rotation', pairs.quaternion())
psim.add_property('edge_length', pairs.vector())

ntypes = 2
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'stiffness', pairs.real())
psim.add_feature_property('type', 'damping_norm', pairs.real())
psim.add_feature_property('type', 'damping_tan', pairs.real())
psim.add_feature_property('type', 'friction', pairs.real())

# psim.set_domain_partitioner(pairs.block_forest())
psim.set_domain_partitioner(pairs.regular_domain_partitioner())
psim.pbc([True, True, False])
psim.build_cell_lists()

psim.compute(update_mass_and_inertia, symbols={'infinity': math.inf })

# 'compute_globals' enables computation of forces on global bodies
psim.compute(spring_dashpot, compute_globals=True)
psim.compute(euler, parameters={'dt': pairs.real()})

gravity_SI = 9.81
psim.compute(gravity, symbols={'gravity_SI': gravity_SI })

psim.generate()

