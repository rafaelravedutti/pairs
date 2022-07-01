#pragma once

namespace pairs {

void *device_alloc(size_t size) { return NULL; }
void *device_realloc(void *ptr, size_t size) { return NULL; }
void copy_to_device(void *h_ptr, const void *d_ptr, size_t count) {}
void copy_to_host(void *d_ptr, const void *h_ptr, size_t count) {}
int atomic_add(int *addr, int val) {
    *addr += val;
    return *addr - val;
}

int atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    const int add_res = *addr + val;
    if(add_res >= capacity) {
        *resize = add_res;
        return *addr;
    }

    return atomic_add(addr, val);
}

}
