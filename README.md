# P4IRS - Parallel and Performance-Portable Particles Intermediate Representation and Simulator

P4IRS is an open-source, stand-alone compiler and domain-specific language for particle simulations which aims at generating optimized code for different target hardwares.
It is released as a Python package and allows users to define kernels, integrators and other particle routines in a high-level and straightforward fashion without the need to implement any backend code.

## Build instructions

There is a Makefile which contains configurable environment variables such as `TESTCASE` compiler parameters evaluate P4IRS performance on different scenarios.
`TESTCASE` refers to any of the files within the `examples` directory, such as `md` and `dem`.

## Usage

To use P4IRS, it is necessary to install it as a Python package and import it with:

```python
import pairs
```

Particle interactions and specific routines to update each particle individually are defined through Python methods. These can make use of defined properties in the simulation, parameters passed in the `compute` method and intrinsic methods from P4IRS:

```python
def lennard_jones(i, j):
    sr2 = 1.0 / squared_distance(i, j)
    sr6 = sr2 * sr2 * sr2 * sigma6[i, j]
    apply(force, delta(i, j) * (48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[i, j]))


def initial_integrate(i):
    linear_velocity[i] += (dt * 0.5) * force[i] / mass[i]
    position[i] += dt * linear_velocity[i]


def final_integrate(i):
    linear_velocity[i] += (dt * 0.5) * force[i] / mass[i]
```

After defining the methods, it is necessary to setup the P4IRS simulations:

```python
dt = 0.005
cutoff_radius = 2.5
skin = 0.3
ntypes = 4
sigma = 1.0
epsilon = 1.0
sigma6 = sigma ** 6
nx = 32
ny = 32
nz = 32
rho = 0.8442
temp = 1.44

# Simulation setup
psim = pairs.simulation(
  "md", # Simulation identifier
  [pairs.point_mass()], # List of shapes
  timesteps=200, # Number of time-steps
  double_prec=True) # Use double-precision

# Particle properties
psim.add_position('position')
psim.add_property('mass', pairs.real(), 1.0)
psim.add_property('velocity', pairs.vector())
psim.add_property('force', pairs.vector(), volatile=True)

# Features and their properties
psim.add_feature('type', ntypes)
psim.add_feature_property('type', 'epsilon', pairs.real(), [epsilon for i in range(ntypes * ntypes)])
psim.add_feature_property('type', 'sigma6', pairs.real(), [sigma6 for i in range(ntypes * ntypes)])

# Simulation domain and initial state
psim.copper_fcc_lattice(nx, ny, nz, rho, temp, ntypes)
psim.set_domain_partitioner(pairs.regular_domain_partitioner())
psim.compute_thermo(100)
```

Then, define the optimization strategies and visualization settings to use:

```python
# Optimization settings
psim.reneighbor_every(20)
psim.compute_half()
psim.build_neighbor_lists(cutoff_radius + skin)
psim.vtk_output("output/md", every=20)
```

Then, all defined particle routines defined must be scheduled for computation:

```python
# Kernels to compute
psim.compute(lennard_jones, cutoff_radius)
psim.compute(euler, symbols={'dt': dt})
```

And finally, it is necessary to define the target and trigger the code generator:

```python
# Target hardware
if target == 'gpu':
  psim.target(pairs.target_gpu())
else:
  psim.target(pairs.target_cpu())

psim.generate()
```

## Citations

TBD

## Credits

P4IRS is developed by the Erlangen National High Performance Computing Center
([NHR@FAU](https://hpc.fau.de/)) at the University of Erlangen-Nürnberg.

## License

[MIT](https://i10git.cs.fau.de/software/pairs/-/blob/master/LICENSE)
