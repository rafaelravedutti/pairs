#include <cstring>
//---
#include "device.hpp"

namespace pairs {

void *device_alloc(size_t) { return nullptr; }
void *device_realloc(void *, size_t) { return nullptr; }
void device_free(void *) {}
void device_synchronize() {}
void copy_to_device(void const *, void *, size_t) {}
void copy_to_host(void const *, void *, size_t) {}
void copy_slice_to_device(void const *, void *, size_t , size_t) {}
void copy_slice_to_host(void const *, void *, size_t , size_t) {}
void copy_static_symbol_to_device(void *, const void *, size_t) {}
void copy_static_symbol_to_host(void *, const void *, size_t) {}

void copy_in_device(void *d_ptr1, const void *d_ptr2, size_t count) {
    std::memcpy(d_ptr1, d_ptr2, count);
}

int atomic_add(int *addr, int val) {
    return host_atomic_add(addr, val);
}

real_t atomic_add(real_t *addr, real_t val) {
    return host_atomic_add(addr, val);
}

int atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    return host_atomic_add_resize_check(addr, val, resize, capacity);
}

}
