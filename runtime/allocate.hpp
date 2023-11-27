#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#pragma once

#ifndef ALIGNMENT
#   define ALIGNMENT    64
#endif

namespace pairs {

inline void *host_alloc(size_t bytesize) {
    void *ptr;
    int errorCode;

    errorCode = posix_memalign(&ptr, ALIGNMENT, bytesize);
    if(errorCode == EINVAL) {
        fprintf(stderr, "Error: Alignment parameter is not a power of two\n");
        exit(EXIT_FAILURE);
    }

    if(errorCode == ENOMEM) {
        fprintf(stderr, "Error: Insufficient memory to fulfill the request\n");
        exit(EXIT_FAILURE);
    }

    if(ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

inline void *host_realloc(void *ptr, size_t new_bytesize, size_t old_bytesize) {
    void *newarray = pairs::host_alloc(new_bytesize);
    if(ptr != NULL) {
        memcpy(newarray, ptr, old_bytesize);
        free(ptr);
    }

    return newarray;
}

}
