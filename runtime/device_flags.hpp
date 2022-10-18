#include <cmath>
#include <iostream>

#pragma once

namespace pairs {

class DeviceFlags {
private:
    unsigned long long int *hflags;
    unsigned long long int *dflags;
    int narrays;
    int nflags;
    static const int narrays_per_flag = 64;
public:
    DeviceFlags(int narrays_) : narrays(narrays_) {
        nflags = std::ceil((double) narrays_ / (double) narrays_per_flag);
        hflags = new unsigned long long int[nflags];
        dflags = new unsigned long long int[nflags];

        for(int i = 0; i < nflags; ++i) {
            hflags[i] = 0xfffffffffffffffful;
            dflags[i] = 0x0;
        }
    }

    inline bool isHostFlagSet(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        return (hflags[flag_index] & (1 << bit)) != 0;
    }

    inline void setHostFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        hflags[flag_index] |= 1 << bit;
    }

    inline void clearHostFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        hflags[flag_index] &= ~(1 << bit);
    }

    inline bool isDeviceFlagSet(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        return (dflags[flag_index] & (1 << bit)) != 0;
    }

    inline void setDeviceFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        dflags[flag_index] |= 1 << bit;
    }

    inline void clearDeviceFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        dflags[flag_index] &= ~(1 << bit);
    }

    void printFlags() {
        std::cerr << "hflags = ";
        for(int i = 0; i < nflags; i++) {
            for(int b = 63; b >= 0; b--) {
                if(hflags[i] & (1 << b)) {
                    std::cerr << "1";
                } else {
                    std::cerr << "0";
                }
            }
        }

        std::cerr << std::endl << "dflags = ";
        for(int i = 0; i < nflags; i++) {
            for(int b = 63; b >= 0; b--) {
                if(dflags[i] & (1 << b)) {
                    std::cerr << "1";
                } else {
                    std::cerr << "0";
                }
            }
        }

        std::cerr << std::endl;
    }

    ~DeviceFlags() {
        delete[] hflags;
        delete[] dflags;
    }
};

}
