#include <iostream>
#include <mpi.h>

#pragma once

namespace pairs {

#ifdef PAIRS_TARGET_CUDA
    #define PAIRS_ATTR_HOST __host__ 
    #define PAIRS_ATTR_DEVICE __device__ 
    #define PAIRS_ATTR_HOST_DEVICE __host__ __device__
#else
    #define PAIRS_ATTR_HOST
    #define PAIRS_ATTR_DEVICE
    #define PAIRS_ATTR_HOST_DEVICE
#endif

namespace flags{
    constexpr int INFINITE = 1 << 0 ;
    constexpr int GHOST    = 1 << 1 ;
    constexpr int FIXED    = 1 << 2 ;
    constexpr int GLOBAL   = 1 << 3 ;
}

enum Shapes {
    Sphere = 0,
    Halfspace = 1,
    PointMass = 2,
    Box = 3
};

//#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
//#else
//typedef float real_t;
//#endif

typedef uint64_t id_t;
typedef int array_t;
typedef int property_t;
typedef int layout_t;
typedef int action_t;

enum PropertyType {
    Prop_Invalid = -1,
    Prop_Integer = 0,
    Prop_UInt64,
    Prop_Real,
    Prop_Vector,
    Prop_Matrix,
    Prop_Quaternion
};

constexpr size_t get_proptype_size(PropertyType type){
    switch (type) {
        case pairs::Prop_Integer:       return sizeof(int);
        case pairs::Prop_UInt64:        return sizeof(uint64_t);
        case pairs::Prop_Real:          return sizeof(real_t);
        case pairs::Prop_Vector:        return 3*sizeof(real_t);
        case pairs::Prop_Matrix:        return 9*sizeof(real_t);
        case pairs::Prop_Quaternion:    return 4*sizeof(real_t);
        default:             return 0;
    }
}

enum DataLayout {
    Invalid = -1,
    AoS = 0,
    SoA
};

enum Actions {
    NoAction = 0,
    ReadAfterWrite = 1,
    WriteAfterRead = 2,
    ReadOnly = 3,
    WriteOnly = 4,
    Ignore = 5
};

enum TimerMarkers {
    MPI = 0,
    DeviceTransfers = 1,
    Offset = 2
};

enum DomainPartitioners {
    RegularPartitioning = 0,
    RegularXYPartitioning = 1,
    BlockForestPartitioning = 2
};

enum LoadBalancingAlgorithms {
    Morton = 0,
    Hilbert = 1,
    Metis = 2,
    Diffusive = 3
};

constexpr const char* getAlgorithmName(LoadBalancingAlgorithms alg) {
    switch (alg) {
        case Morton:    return "Morton";
        case Hilbert:   return "Hilbert";
        case Metis:     return "Metis";
        case Diffusive: return "Diffusive";
        default:        return "Invalid";
    }
}

#ifdef DEBUG
#   include <assert.h>
#   define PAIRS_DEBUG(...)     {                                                   \
                                    int __init_flag;                                \
                                    int __rank;                                     \
                                    MPI_Initialized(&__init_flag);                  \
                                    if(__init_flag == 0) {                          \
                                       fprintf(stderr, __VA_ARGS__);                \
                                    } else {                                        \
                                        MPI_Comm_rank(MPI_COMM_WORLD, &__rank);     \
                                        if(__rank == 0) {                           \
                                            fprintf(stderr, __VA_ARGS__);           \
                                        }                                           \
                                    }                                               \
                                }

#   define PAIRS_ASSERT(a)      assert(a)
#   define PAIRS_EXCEPTION(a)
#else
// #   define PAIRS_DEBUG(...) {printf(__VA_ARGS__);}
#   define PAIRS_DEBUG(...)
#   define PAIRS_ASSERT(a)
#   define PAIRS_EXCEPTION(a)
#endif

#define PAIRS_ERROR(...)        fprintf(stderr, __VA_ARGS__)
#define MIN(a,b)                ((a) < (b) ? (a) : (b))
#define MAX(a,b)                ((a) > (b) ? (a) : (b))
#define SIGN(a)                 ((a) < 0 ? -1 : 1)

}