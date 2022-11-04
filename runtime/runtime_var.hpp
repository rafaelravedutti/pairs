#include "devices/device.hpp"

#pragma once

namespace pairs {

template<typename T>
class RuntimeVar{
protected:
    T *h_ptr, *d_ptr;

public:
    RuntimeVar(T *ptr) {
        h_ptr = ptr;
        d_ptr = (T *) pairs::device_alloc(sizeof(T));
    }

    ~RuntimeVar() {
        pairs::device_free(d_ptr);
    }

    inline void copyToDevice() { pairs::copy_to_device(h_ptr, d_ptr, sizeof(T)); }
    inline void copyToHost() { pairs::copy_to_host(d_ptr, h_ptr, sizeof(T)); }
    inline T *getHostPointer() { return h_ptr; }
    inline T *getDevicePointer() { return d_ptr; }
};

}
