# P4IRS - Parallel and Performance-Portable Particles Intermediate Representation and Simulator

P4IRS is an open-source, stand-alone compiler and domain-specific language for particle simulations, which aims at generating optimized code for parallel execution on multi-CPU and multi-GPU platforms.

P4IRS allows users to define kernels, integrators and other particle routines in a high-level and straightforward fashion using simple Python methods. It then generates C++ or CUDA code along with interface classes that allow the user to interact with the code and access the particle data on host and device. P4IRS can be used as a stand-alone framework, or it can be integrated as a library within other projects for complex multiphysics simulations. 


## Usage

### Python Interface

Start by importing the library and creating a simulation instance:

```python
import pairs
psim = pairs.simulation("dem", double_prec=True)
```
---
Register properties (their name, type and communication mode):

Communication mdoes:

- `always`: Communicated during `updateDomain()`, `reneighbor()`, and `refreshGhosts()`  
- `never`: Never communicated
- `on_reneighbor`: Only communicated during `updateDomain()` and `reneighbor()` 
- `on_reduction`: Only communicated in reverse and then reduced during `reduceGhosts()` 

Shape registration:

```python
psim.add_shape(pairs.sphere('radius'))
psim.add_shape(pairs.halfspace('normal'))
```

Example for property registration:

```python
psim.add_position('position')
psim.add_property('radius', pairs.real(), pairs.on_reneighbor())
psim.add_property('linear_velocity', pairs.vector(), pairs.always())
psim.add_property('force', pairs.vector(), pairs.never())
```

---

Features & Feature-Properties:
Features represent subsets of particles (e.g., material types), while feature-properties are pairswise properties between these subsets (e.g., contact stiffness).

```python
psim.add_feature('type', nkinds=2)  # 2 types of materials
psim.add_feature_property('type', 'stiffness', pairs.real(), [1e7, 1e5, 1e5, 1e4])  # A 2x2 lookup tabel for stifnesses
```

Or define values at runtime using the accessor:

```cpp
ac.setTypeStiffness(0, 1, 1e5)  // C++
```

---

Select the domain partitioner and PBCs:

```python
psim.set_domain_partitioner(pairs.block_forest()) # or pairs.regular_domain_partitioner()
psim.pbc([True, True, False])  # PBCs in x and y
```

---

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

---

Acceleration Structures & Optimizations

```python
psim.build_cell_lists(store_neighbors_per_cell=True, use_halo_cells=False)
# psim.build_neighbor_lists()  # Verlet list for MD
```

---

Trigger code generation:

```python
psim.generate()
```

Compilation is handled by CMake, which will compile the generated files with the appropriate backend.

---

### C++ Interface

See [examples](examples/modular).

## Build Instructions

P4IRS can be built in two different modes using the CMake build system. Before we demostrate each mode, ensure you have CMake, MPI and CUDA (if targeting GPU execution) available in your environment.

In the following, we assume we have created and navigated to a build directory: `mkdir build; cd build` 

**Basic CMake flags:**  
* Pass your input script to CMake using `-DPAIRS_INPUT_SCRIPT=path/to/script.py`  
* Enable CUDA with `-DPAIRS_BUILD_WITH_CUDA=ON`
* Enable waLBerla support with `-DPAIRS_BUILD_WITH_WALBERLA=ON` for using BlockForest domain partitioning and dynamic load balancing


### 1. Stand-Alone P4IRS Application
---------------------
To build a C++ application using P4IRS, provide the list of your source files to CMake using the `-DPAIRS_INPUT_SRCS` flag (semicolon-seperated).

**Example**: Build the application [sd_1.cpp](examples/modular/sd_1.cpp) using [spring_dashpot.py](examples/modular/spring_dashpot.py) as the input script.

```
cmake -DPAIRS_INPUT_SCRIPT=../examples/modular/spring_dashpot.py -DPAIRS_INPUT_SRCS=../examples/modular/sd_1.cpp -DPAIRS_BUILD_WITH_WALBERLA=ON ..
```
Now call `make` and an **executable** is built.


### 2. P4IRS as a Library
---------------------
P4IRS can also be compiled as a library for integration into larger projects.  
To compile P4IRS as a library, simply do not pass any `PAIRS_INPUT_SRCS` to CMake. Configure CMake and call `make` as usual, and a **static library** is built. You can then include P4IRS and its dependencies in your build system as follows:
```cmake
find_package(pairs REQUIRED HINTS "path/to/pairs/build" NO_DEFAULT_PATH)
target_include_directories(my_app PUBLIC ${PAIRS_INCLUDE_DIRS})
target_link_libraries(my_app PUBLIC ${PAIRS_LIBRARIES})
```

## Citations

TBD

## Credits

P4IRS is developed by the Erlangen National High Performance Computing Center
([NHR@FAU](https://hpc.fau.de/)) at the University of Erlangen-Nürnberg.

## License

[MIT](https://i10git.cs.fau.de/software/pairs/-/blob/master/LICENSE)
