#include <iostream>

#pragma once

//#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
//#else
//typedef float real_t;
//#endif

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

enum DomainPartitioning {
    DimRanges = 0,
    BoxList,
};

#ifdef DEBUG
#   include <assert.h>
#   define PAIRS_DEBUG(...)     fprintf(stderr, __VA_ARGS__)
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
