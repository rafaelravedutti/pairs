#include "../pairs.hpp"

namespace pairs_host_interface {

int get_uid(int *uid, int i) { return uid[i]; }
int get_shape(int *shape, int i) { return shape[i]; }
int get_flags(int *flags, int i) { return flags[i]; }
double get_position(double *position, int i, int j, int capacity) { return position[i * 3 + j]; }
double get_mass(double *mass, int i) { return mass[i]; }
double get_linear_velocity(double *linear_velocity, int i, int j, int capacity) { return linear_velocity[i * 3 + j]; }
double get_force(double *force, int i, int j, int capacity) { return force[i * 3 + j]; }
int get_type(int *type, int i) { return type[i]; }

}

namespace pairs_cuda_interface {

__inline__ __device__ int get_uid(int *uid, int i) { return uid[i]; }
__inline__ __device__ int get_shape(int *shape, int i) { return shape[i]; }
__inline__ __device__ int get_flags(int *flags, int i) { return flags[i]; }
__inline__ __device__ double get_position(double *position, int i, int j, int capacity) { return position[i * 3 + j]; }
__inline__ __device__ double get_mass(double *mass, int i) { return mass[i]; }
__inline__ __device__ double get_linear_velocity(double *linear_velocity, int i, int j, int capacity) { return linear_velocity[i * 3 + j]; }
__inline__ __device__ double get_force(double *force, int i, int j, int capacity) { return force[i * 3 + j]; }
__inline__ __device__ int get_type(int *type, int i) { return type[i]; }

}
