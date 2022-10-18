#include <iostream>

#pragma once

typedef double real_t;
typedef int array_t;
typedef int property_t;
typedef int layout_t;

enum PropertyType {
    Prop_Invalid = -1,
    Prop_Integer = 0,
    Prop_Float,
    Prop_Vector
};

enum DataLayout {
    Invalid = -1,
    AoS = 0,
    SoA
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