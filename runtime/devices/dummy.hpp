#pragma once

namespace pairs {

void *device_alloc(size_t size) { return NULL; }
void *device_realloc(void *ptr, size_t size) { return NULL; }
void copy_to_device(void *h_ptr, const void *d_ptr, size_t count) {}
void copy_to_host(void *d_ptr, const void *h_ptr, size_t count) {}

}
