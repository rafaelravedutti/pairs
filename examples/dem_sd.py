import math
import pairs
import sys

def update_mass_and_inertia(i):
    rotation_matrix[i] = diagonal_matrix(1.0)
    rotation[i] = default_quaternion()

    if is_sphere(i):
        inv_inertia[i] = inversed(diagonal_matrix(0.4 * mass[i] * radius[i] * radius[i]))

    else:
        mass[i] = infinity
        inv_inertia[i] = 0.0


def linear_spring_dashpot(i, j):
    delta_ij = -penetration_depth(i, j)
    skip_when(delta_ij < 0.0)

    meff = 1.0 / ((1.0 / mass[i]) + (1.0 / mass[j]))
    stiffness_norm = meff * (pi * pi + lnDryResCoeff * lnDryResCoeff) / \
                     (collisionTime_SI * collisionTime_SI)
    damping_norm = -2.0 * meff * lnDryResCoeff / collisionTime_SI
    damping_tan = sqrt(kappa) * damping_norm

    velocity_wf_i = linear_velocity[i] + cross(angular_velocity[i], contact_point(i, j) - position[i])
    velocity_wf_j = linear_velocity[j] + cross(angular_velocity[j], contact_point(i, j) - position[j])

    rel_vel = -(velocity_wf_i - velocity_wf_j)
    rel_vel_n = dot(rel_vel, contact_normal(i, j))
    rel_vel_t = rel_vel - rel_vel_n * contact_normal(i, j)
    fNabs = stiffness_norm * delta_ij + damping_norm * rel_vel_n
    fN = fNabs * contact_normal(i, j)

    fTabs = min(damping_tan * length(rel_vel_t), friction_dynamic[i, j] * fNabs) # static or dynamic?
    fT = fTabs * normalized(rel_vel_t)

    partial_force = fN + fT

    apply(force, partial_force)
    apply(torque, cross(contact_point(i, j) - position, partial_force))


def euler(i):
    inv_mass = 1.0 / mass[i]
    position[i] += 0.5 * inv_mass * force[i] * dt * dt + linear_velocity[i] * dt
    linear_velocity[i] += inv_mass * force[i] * dt
    wdot = rotation_matrix[i] * (inv_inertia[i] * torque[i]) * transposed(rotation_matrix[i])
    phi = angular_velocity[i] * dt + 0.5 * wdot * dt * dt
    rotation[i] = quaternion(phi, length(phi)) * rotation[i]
    rotation_matrix[i] = quaternion_to_rotation_matrix(rotation[i])
    angular_velocity[i] += wdot * dt


def gravity(i):
    volume = (4.0 / 3.0) * pi * radius[i] * radius[i] * radius[i]
    force[i][2] = force[i][2] - (densityParticle_SI - densityFluid_SI) * volume * gravity_SI
    # if type[i]==0:
    #     printf("gravity", i, volume, force[i][2], volume/2, 2>1)


cmd = sys.argv[0]
target = sys.argv[1] if len(sys.argv[1]) > 1 else "none"
if target != 'cpu' and target != 'gpu':
    print(f"Invalid target, use {cmd} <cpu/gpu>")

# Config file parameters
domainSize_SI=[0.1, 0.1, 0.1]
diameter_SI = 0.009
gravity_SI = 9.81
densityFluid_SI = 1000
densityParticle_SI = 2550
generationSpacing_SI = 0.01
initialVelocity_SI = 1
dt_SI = 5e-5
frictionCoefficient = 0.5
restitutionCoefficient = 0.1
collisionTime_SI = 5e-4
poissonsRatio = 0.22
timeSteps = 10000
visSpacing = 200
denseBottomLayer = False
bottomLayerOffsetFactor = 1.0
kappa = 2.0 * (1.0 - poissonsRatio) / (2.0 - poissonsRatio) # from Thornton et al

minDiameter_SI = diameter_SI #* 0.9
maxDiameter_SI = diameter_SI #* 1.1
linkedCellWidth = 1.01 * maxDiameter_SI

ntypes = 1
lnDryResCoeff = math.log(restitutionCoefficient)
frictionStatic = 0.0
frictionDynamic = frictionCoefficient

psim = pairs.simulation(
    "dem_sd",
    [pairs.sphere(), pairs.halfspace()],
    timesteps=timeSteps,
    double_prec=True,
    use_contact_history=False,
    particle_capacity=1000000,
    neighbor_capacity=20,
    debug=True, generate_whole_program=False)

if target == 'gpu':
    psim.target(pairs.target_gpu())
else:
    psim.target(pairs.target_cpu())
    #psim.target(pairs.target_cpu(parallel=True))

psim.add_position('position')
psim.add_property('mass', pairs.real(), 1.0)
psim.add_property('linear_velocity', pairs.vector())
psim.add_property('angular_velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)
psim.add_property('torque', pairs.vector(), volatile=True)
psim.add_property('hydrodynamic_force', pairs.vector())
psim.add_property('hydrodynamic_torque', pairs.vector())
psim.add_property('old_hydrodynamic_force', pairs.vector())
psim.add_property('old_hydrodynamic_torque', pairs.vector())
psim.add_property('radius', pairs.real(), 1.0)
psim.add_property('normal', pairs.vector())
psim.add_property('inv_inertia', pairs.matrix())
psim.add_property('rotation_matrix', pairs.matrix())
psim.add_property('rotation', pairs.quaternion())
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'friction_static', pairs.real(), [frictionStatic for i in range(ntypes * ntypes)])
psim.add_feature_property('type', 'friction_dynamic', pairs.real(), [frictionDynamic for i in range(ntypes * ntypes)])

psim.set_domain([0.0, 0.0, 0.0, domainSize_SI[0], domainSize_SI[1], domainSize_SI[2]])
# psim.set_domain_partitioner(pairs.block_forest(), initDomainFromWalberla=True)
psim.set_domain_partitioner(pairs.block_forest())
# psim.set_domain_partitioner(pairs.regular_domain_partitioner_xy())
psim.pbc([False, False, False])
psim.dem_sc_grid(
    domainSize_SI[0], domainSize_SI[1], domainSize_SI[2]/2, generationSpacing_SI,
    diameter_SI, minDiameter_SI, maxDiameter_SI, initialVelocity_SI, densityParticle_SI, ntypes)

#psim.read_particle_data(
#    "data/spheres.input",
#    "data/spheres_4x4x2.input",
#    "data/spheres_6x6x2.input",
#    "data/spheres_8x8x2.input",
#    ['uid', 'type', 'mass', 'radius', 'position', 'linear_velocity', 'flags'],
#    pairs.sphere())

#psim.read_particle_data(
#    "data/spheres_bottom.input",
#    ['type', 'mass', 'radius', 'position', 'linear_velocity', 'flags'],
#    pairs.sphere())

# psim.read_particle_data(
#     "data/planes.input",
#     ['type', 'mass', 'position', 'normal', 'flags'],
#     pairs.halfspace())

psim.setup(update_mass_and_inertia, {'densityParticle_SI': densityParticle_SI,
                                     'pi': math.pi,
                                     'infinity': math.inf })

#psim.compute_half()
psim.build_cell_lists(linkedCellWidth)
psim.vtk_output(f"output/dem_{target}", frequency=visSpacing)

psim.compute(gravity,
             symbols={'densityParticle_SI': densityParticle_SI,
                      'densityFluid_SI': densityFluid_SI,
                      'gravity_SI': gravity_SI,
                      'pi': math.pi })

psim.compute(linear_spring_dashpot,
             linkedCellWidth,
             symbols={'dt': dt_SI,
                      'pi': math.pi,
                      'kappa': kappa,
                      'lnDryResCoeff': lnDryResCoeff,
                      'collisionTime_SI': collisionTime_SI})

psim.compute(euler, symbols={'dt': dt_SI})
psim.generate()
