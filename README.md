# P4IRS - Parallel and Performance-Portable Particles Intermediate Representation and Simulator

P4IRS is an open-source, stand-alone compiler and domain-specific language for particle simulations, which aims at generating optimized code for parallel execution on multi-CPU and multi-GPU platforms.

P4IRS allows users to define kernels, integrators and other particle routines in a high-level and straightforward fashion using simple Python methods. It then generates C++ or CUDA code along with interface classes that allow the user to interact with the code and access the particle data on host and device. P4IRS can be used as a stand-alone framework, or it can be integrated as a library within other projects for complex multiphysics simulations. 


## Usage

### Python Interface

Start by importing the library and creating a simulation instance:

```python
import pairs
psim = pairs.Simulation()
```

Register the shapes present in the simulation:

```python
psim.add_shape(pairs.sphere('radius'))
psim.add_shape(pairs.halfspace('normal'))
```

Register properties (name, type and communication mode):

```python
psim.add_position('position')
psim.add_property('radius', pairs.real(), pairs.on_reneighbor())
psim.add_property('normal', pairs.vector(), pairs.on_reneighbor())
psim.add_property('linear_velocity', pairs.vector(), pairs.always())
psim.add_property('force', pairs.vector(), pairs.never())
```
Communication modes for properties:

| Mode            | Description |
|-----------------|-------------|
| `always`        | Communicated during `updateDomain()`, `reneighbor()`, and `refreshGhosts()` |
| `on_reneighbor` | Communicated during `updateDomain()` and `reneighbor()` |
| `on_reduction`  | Communicated in reverse for ghost particles and reduced during `reduceGhosts()` |
| `never`         | Never communicated |

Features & Feature-Properties:
Features represent subsets of particles (e.g., material types), while feature-properties are pairswise properties between these subsets (e.g., contact stiffness).

```python
psim.add_feature('type', nkinds=2)  # 2 types of materials
psim.add_feature_property('type', 'stiffness', pairs.real(), [1e7, 1e5, 1e5, 1e4])  # A 2x2 lookup tabel for stifnesses
```

Select the domain partitioner and PBCs:

```python
psim.set_domain_partitioner(pairs.block_forest()) # or pairs.regular_domain_partitioner()
psim.pbc([True, True, False])  # PBCs in x and y
```

Define kernels as simple Python methods.

One-Body Kernel:

```python
def gravity(i):
    force[i][2] -= mass[i] * gravity_SI

```

Two-Body Kernel:

```python
def spring_dashpot(i, j):
    rel_vel = linear_velocity[i] - linear_velocity[j]
    rel_vel_n = dot(rel_vel, contact_normal(i, j))
    f_spring = - stiffness[i, j] * penetration_depth(i, j) * contact_normal(i, j)
    f_damping = - damping[i, j] * rel_vel_n
    apply(force, f_spring + f_damping)
```

Register with optional features:

```python
psim.compute(gravity, symbols={'gravity_SI': 9.81}) # symbols are known at compile-time. Use `parameters` for runtime definition
psim.compute(spring_dashpot, compute_globals=True, run_on_device=True, profile=True)
```

Acceleration Structures & Optimizations:

```python
psim.build_cell_lists(store_neighbors_per_cell=True, use_halo_cells=False)
# psim.build_neighbor_lists()  # Verlet list for MD
```

Finally, trigger code generation:

```python
psim.generate()
```
---

### C++ Interface

See [examples](apps/examples).

## Build Instructions

P4IRS applications are built using CMake. Ensure that CMake and MPI are available on your system. CUDA is required only when targeting GPU execution.

### CMake Integration

* **Create your CMake target** 
  
For example, create an executable from your source file:

```cmake
add_executable( MyApp path/to/main.cpp )
```

* **Generate the P4IRS library** 

Use the CMake function `pairs_generate_lib` to generate a P4IRS library from your Python script:

```cmake
pairs_generate_lib(
    GEN_LIB     MyPairsLib
    SCRIPT      path/to/my_pairs_script.py
)
```
This generates the P4IRS code and compiles it into a static library called `MyPairsLib` that includes all necessary dependencies. As with any CMake target, the name of the library must be unique within your project.

* **Link your CMake target to the P4IRS library** 

Aattach the library to your executable:

```cmake
target_link_libraries( MyApp MyPairsLib )
```
Multiple targets may link against the same generated P4IRS library, reusing the same generated code.

### Configure and Build

**Basic CMake flags:**  
| Option                          | Description |
|---------------------------------|-------------|
| `PAIRS_BUILD_WITH_CUDA`     | Enable CUDA support |
| `PAIRS_BUILD_WITH_WALBERLA` | Enable support for waLBerla BlockForest domain partitioning and dynamic load balancing |
| `PAIRS_BUILD_WITH_LIKWID`   | Enable profiling compute kernels with LIKWID performance tools |
| `PAIRS_BUILD_EXAMPLES`      | Enable building the example apps |
| `PAIRS_BUILD_BENCHMAKRS`    | Enable building the benchmark apps |


**Example**: Build the application [sd_4.cpp](apps/examples/SD/sd_4.cpp) for GPU execution, using the CUDA code generated by [spring_dashpot.py](apps/examples/SD/spring_dashpot.py).

```
cmake -S . -B build -DPAIRS_BUILD_EXAMPLES=ON -DPAIRS_BUILD_WITH_WALBERLA=ON -DPAIRS_BUILD_WITH_CUDA=ON
cmake --build build --target sd_4 -j
``` 
and then run it for example using `srun` on 8 GPUs:
```
srun -n 8 --gpus-per-task=1 ./sd_4
```


## Citations

Ravedutti Lucio Machado, R., Eitzinger, J., & Köstler, H. (2025). *P4IRS: An intermediate representation and compiler for parallel and performance-portable particle simulations*. The International Journal of High Performance Computing Applications. https://doi.org/10.1177/10943420251405928

## Credits

P4IRS is developed by the Erlangen National High Performance Computing Center
([NHR@FAU](https://hpc.fau.de/)) at the University of Erlangen-Nürnberg.

## License

[MIT](https://i10git.cs.fau.de/software/pairs/-/blob/master/LICENSE)
