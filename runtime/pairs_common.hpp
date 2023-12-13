#include <iostream>
#include <mpi.h>

#pragma once

//#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
//#else
//typedef float real_t;
//#endif

#ifndef PAIRS_TARGET_CUDA
#   ifdef ENABLE_CUDA_AWARE_MPI
#       undef ENABLE_CUDA_AWARE_MPI
#   endif
#endif

typedef int array_t;
typedef int property_t;
typedef int layout_t;
typedef int action_t;

enum PropertyType {
    Prop_Invalid = -1,
    Prop_Integer = 0,
    Prop_Real,
    Prop_Vector,
    Prop_Matrix,
    Prop_Quaternion
};

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

enum Timers {
    All = 0,
    Communication = 1,
    DeviceTransfers = 2,
    Offset = 3
};

enum DomainPartitioners {
    Regular = 0,
    RegularXY = 1,
    BoxList = 2,
};

#ifdef DEBUG
#   include <assert.h>
#   define PAIRS_DEBUG(...)     {                                                   \
                                    int __rank;                                     \
                                    MPI_Comm_rank(MPI_COMM_WORLD, &__rank);         \
                                    if(__rank == 0) {                               \
                                       fprintf(stderr, __VA_ARGS__);                \
                                    }                                               \
                                }

#   define PAIRS_ASSERT(a)      assert(a)
#   define PAIRS_EXCEPTION(a)
#else
#   define PAIRS_DEBUG(...)
#   define PAIRS_ASSERT(a)
#   define PAIRS_EXCEPTION(a)
#endif

#define PAIRS_ERROR(...)        fprintf(stderr, __VA_ARGS__)
#define MIN(a,b)                ((a) < (b) ? (a) : (b))
#define MAX(a,b)                ((a) > (b) ? (a) : (b))
