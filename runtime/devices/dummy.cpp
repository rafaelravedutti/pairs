#include <cstring>
//---
#include "device.hpp"

namespace pairs {

void *device_alloc(size_t size) { return nullptr; }
void *device_realloc(void *ptr, size_t size) { return nullptr; }
void device_free(void *ptr) {}
void device_synchronize() {}
void copy_to_device(void const *h_ptr, void *d_ptr, size_t count) {}
void copy_to_host(void const *d_ptr, void *h_ptr, size_t count) {}
void copy_slice_to_device(void const *h_ptr, void *d_ptr, size_t offset, size_t count) {}
void copy_slice_to_host(void const *d_ptr, void *h_ptr, size_t offset, size_t count) {}
void copy_static_symbol_to_device(void *h_ptr, const void *d_ptr, size_t count) {}
void copy_static_symbol_to_host(void *d_ptr, const void *h_ptr, size_t count) {}

void copy_in_device(void *d_ptr1, const void *d_ptr2, size_t count) {
    std::memcpy(d_ptr1, d_ptr2, count);
}

}
