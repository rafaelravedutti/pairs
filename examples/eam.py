import pairs
import sys


def linear_spring_dashpot(i, j):
    delta = -penetration_depth
    skip_if(delta < 0.0)

    rel_vel = -velocity_wf[i] - velocity_wf[j]
    rel_vel_n = dot(rel_vel, contact_normal) * contact_normal
    rel_vel_t = rel_vel - rel_vel_n

    fN = stiffness_norm[i, j] * delta * contact_normal + damping_norm[i, j] * relVelN;

    tan_spring_disp = tangential_spring_displacement[i, j]
    impact_vel_magnitude = impact_velocities_magnitude[i, j]
    sticking = is_sticking[i, j]

    rotated_tan_disp = tan_spring_disp - contact_normal * (contact_normal * tan_spring_disp)
    new_tan_spring_disp = select(sq_len(rotated_tan_disp) <= 0.0,
                                 0.0, 
                                 rotated_tan_disp * length(tan_spring_disp) / length(rotated_tan_disp))
    new_tan_spring_disp += dt * rel_vel_t

    fTLS = stiffness_tan[i, j] * new_tan_spring_disp + damping_tan[i, j] * rel_vel_t
    fTLS_len = length(fTLS)
    t = normalized(fTLS)

    f_friction_abs_static = friction_static[i, j] * length(fN)
    f_friction_abs_dynamic = friction_dynamic[i, j] * length(fN)
    tan_vel_threshold = 1e-8

    cond1 = sticking and rel_vel_t_len < tan_vel_threshold and fTLS_len < f_friction_abs_static
    cond2 = sticking and fTLS_len < f_friction_abs_dynamic
    f_friction_abs = select(cond1, f_friction_abs_static, f_friction_abs_dynamic)
    n_sticking = select(cond1 or cond2 or fTLS_len < f_friction_abs_dynamic, True, False)
    n_T_spring_disp = select(not cond1 and not cond2 and stiffness_tan[i, j] > 0.0,
                             (f_friction_abs * t - damping_tan[i, j] * rel_vel_t) / stiffness_tan[i, j],
                             new_tan_spring_disp2)

    tangential_spring_displacement[i, j] = n_T_spring_disp
    impact_velocities_magnitude[i, j] = impact_vel_magnitude
    is_sticking[i, j] = n_sticking

    fTabs = min(fTLS_len, f_friction_abs)
    fT = fTabs * t
    force[i] += fN + fT


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

psim = pairs.simulation("dem", debug=True)
psim.add_position('position')
psim.add_property('mass', Types.Double, 1.0)
psim.add_property('velocity', Types.Vector)
psim.add_property('force', Types.Vector, vol=True)
psim.add_feature('type')
psim.add_feature_property('type', 'stiffness_norm', Types.Double)
psim.add_feature_property('type', 'stiffness_tan', Types.Double)
psim.add_feature_property('type', 'damping_norm', Types.Double)
psim.add_feature_property('type', 'damping_tan', Types.Double)
psim.add_feature_property('type', 'friction_static', Types.Double)
psim.add_feature_property('type', 'friction_dynamic', Types.Double)
psim.add_contact_history_property('is_sticking', Types.Bool, False)
psim.add_contact_history_property('tangential_spring_displacement', Types.Vector, [0.0, 0.0, 0.0])
psim.add_contact_history_property('impact_velocity_magnitude', Types.Double, 0.0)

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
