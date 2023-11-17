import math
import pairs
import sys


def update_mass_and_inertia(i):
    rotation_matrix[i] = diagonal_matrix(1.0)
    rotation_quat[i] = default_quaternion()

    if is_sphere(i):
        #mass[i] = ((4.0 / 3.0) * pi) * radius[i] * radius[i] * radius[i] * densityParticle_SI
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
    stiffness_tan = kappa * stiffness_norm
    damping_norm = -2.0 * meff * lnDryResCoeff / collisionTime_SI
    damping_tan = sqrt(kappa) * damping_norm

    velocity_wf_i = linear_velocity[i] + cross(angular_velocity[i], contact_point(i, j) - position[i])
    velocity_wf_j = linear_velocity[j] + cross(angular_velocity[j], contact_point(i, j) - position[j])

    rel_vel = -(velocity_wf_i - velocity_wf_j)
    rel_vel_n = dot(rel_vel, contact_normal(i, j)) * contact_normal(i, j)
    rel_vel_t = rel_vel - rel_vel_n
    fN = stiffness_norm * delta_ij * contact_normal(i, j) + damping_norm * rel_vel_n;

    tan_spring_disp = tangential_spring_displacement[i, j]
    impact_vel_magnitude = impact_velocity_magnitude[i, j]
    impact_magnitude = select(impact_vel_magnitude > 0.0, impact_vel_magnitude, length(rel_vel))
    sticking = is_sticking[i, j]

    rot_tan_disp = tan_spring_disp - contact_normal(i, j) * dot(tan_spring_disp, contact_normal(i, j))
    rot_tan_disp_len2 = squared_length(rot_tan_disp)
    new_tan_spring_disp = dt * rel_vel_t + \
                          select(rot_tan_disp_len2 <= 0.0,
                                 zero_vector(),
                                 rot_tan_disp * sqrt(squared_length(tan_spring_disp) / rot_tan_disp_len2))

    fTLS = stiffness_tan * new_tan_spring_disp + damping_tan * rel_vel_t
    fTLS_len = length(fTLS)
    t = normalized(fTLS)

    f_friction_abs_static = friction_static[i, j] * length(fN)
    f_friction_abs_dynamic = friction_dynamic[i, j] * length(fN)
    tan_vel_threshold = 1e-8

    cond1 = sticking == 1 and length(rel_vel_t) < tan_vel_threshold and fTLS_len < f_friction_abs_static
    cond2 = sticking == 1 and fTLS_len < f_friction_abs_dynamic
    f_friction_abs = select(cond1, f_friction_abs_static, f_friction_abs_dynamic)
    n_sticking = select(cond1 or cond2 or fTLS_len < f_friction_abs_dynamic, 1, 0)

    if not cond1 and not cond2 and stiffness_tan > 0.0:
        tangential_spring_displacement[i, j] = (f_friction_abs * t - damping_tan * rel_vel_t) / stiffness_tan

    else:
        tangential_spring_displacement[i, j] = new_tan_spring_disp

    impact_velocity_magnitude[i, j] = impact_magnitude
    is_sticking[i, j] = n_sticking
    contact_used[i][j] = 1

    fTabs = min(fTLS_len, f_friction_abs)
    fT = fTabs * t
    partial_force = fN + fT

    apply(force, partial_force)
    apply(torque, cross(contact_point(i, j) - position, partial_force))


def euler(i):
    inv_mass = 1.0 / mass[i]
    position[i] += 0.5 * inv_mass * force[i] * dt * dt + linear_velocity[i] * dt
    linear_velocity[i] += inv_mass * force[i] * dt
    wdot = rotation_matrix[i] * (inv_inertia[i] * torque[i]) * transposed(rotation_matrix[i])
    phi = angular_velocity[i] * dt + 0.5 * wdot * dt * dt
    rotation_quat[i] = quaternion(phi, length(phi)) * rotation_quat[i]
    rotation_matrix[i] = quaternion_to_rotation_matrix(rotation_quat[i])
    angular_velocity[i] += wdot * dt


def gravity(i):
    volume = (4.0 / 3.0) * pi * radius[i] * radius[i] * radius[i]
    force[i][2] = force[i][2] - (densityParticle_SI - densityFluid_SI) * volume * gravity_SI


cmd = sys.argv[0]
target = sys.argv[1] if len(sys.argv[1]) > 1 else "none"
if target != 'cpu' and target != 'gpu':
    print(f"Invalid target, use {cmd} <cpu/gpu>")

# Config file parameters
domainSize_SI = [0.8, 0.015, 0.2]
blocks = [3, 3, 1]
diameter_SI = 0.0029
gravity_SI = 9.81
densityFluid_SI = 1000
densityParticle_SI = 2550
generationSpacing_SI = 0.005
initialVelocity_SI = 1
dt_SI = 5e-5
frictionCoefficient = 0.5
restitutionCoefficient = 0.1
collisionTime_SI = 5e-4
poissonsRatio = 0.22
timeSteps = 10000
visSpacing = 100
denseBottomLayer = False
bottomLayerOffsetFactor = 1.0
kappa = 2.0 * (1.0 - poissonsRatio) / (2.0 - poissonsRatio) # from Thornton et al

minDiameter_SI = diameter_SI * 0.9
maxDiameter_SI = diameter_SI * 1.1
linkedCellWidth = 1.01 * maxDiameter_SI

skin = 0.0
ntypes = 1

lnDryResCoeff = math.log(restitutionCoefficient);
frictionStatic = 0.0
frictionDynamic = frictionCoefficient

psim = pairs.simulation(
    "dem",
    [pairs.sphere(), pairs.halfspace()],
    timesteps=timeSteps,
    double_prec=True,
    use_contact_history=True)

psim.add_position('position')
psim.add_property('mass', pairs.real(), 1.0)
psim.add_property('linear_velocity', pairs.vector())
psim.add_property('angular_velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)
psim.add_property('torque', pairs.vector(), volatile=True)
psim.add_property('radius', pairs.real(), 1.0)
psim.add_property('normal', pairs.vector())
psim.add_property('inv_inertia', pairs.matrix())
psim.add_property('rotation_matrix', pairs.matrix())
psim.add_property('rotation_quat', pairs.quaternion())
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'friction_static', pairs.real(), [frictionStatic for i in range(ntypes * ntypes)])
psim.add_feature_property('type', 'friction_dynamic', pairs.real(), [frictionDynamic for i in range(ntypes * ntypes)])
psim.add_contact_property('is_sticking', pairs.int32(), 0)
psim.add_contact_property('tangential_spring_displacement', pairs.vector(), [0.0, 0.0, 0.0])
psim.add_contact_property('impact_velocity_magnitude', pairs.real(), 0.0)

psim.set_domain([0.0, 0.0, 0.0, domainSize_SI[0], domainSize_SI[1], domainSize_SI[2]])
psim.pbc([True, True, False])
psim.read_particle_data(
    "data/spheres.input",
    ['uid', 'type', 'mass', 'radius', 'position', 'linear_velocity', 'flags'],
    pairs.sphere())

#psim.read_particle_data(
#    "data/spheres_bottom.input",
#    ['type', 'mass', 'radius', 'position', 'linear_velocity', 'flags'],
#    pairs.sphere())

psim.read_particle_data(
    "data/planes.input",
    ['uid', 'type', 'mass', 'position', 'normal', 'flags'],
    pairs.halfspace())

psim.setup(update_mass_and_inertia, {'densityParticle_SI': densityParticle_SI,
                                     'pi': math.pi,
                                     'infinity': math.inf })

#psim.compute_half()
psim.build_neighbor_lists(linkedCellWidth + skin)
psim.vtk_output(f"output/dem_{target}", frequency=visSpacing)

psim.compute(gravity,
             symbols={'densityParticle_SI': densityParticle_SI,
                      'densityFluid_SI': densityFluid_SI,
                      'gravity_SI': gravity_SI,
                      'pi': math.pi })

psim.compute(linear_spring_dashpot,
             linkedCellWidth + skin,
             symbols={'dt': dt_SI,
                      'pi': math.pi,
                      'kappa': kappa,
                      'lnDryResCoeff': lnDryResCoeff,
                      'collisionTime_SI': collisionTime_SI})

psim.compute(euler, symbols={'dt': dt_SI})

if target == 'gpu':
    psim.target(pairs.target_gpu())
else:
    psim.target(pairs.target_cpu())

psim.generate()
