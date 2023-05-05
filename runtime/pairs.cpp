#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"
#include "devices/device.hpp"
#include "domain/regular_6d_stencil.hpp"

namespace pairs {

void PairsSimulation::initDomain(int *argc, char ***argv, real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
    if(dom_part_type == DimRanges) {
        dom_part = new Regular6DStencil(xmin, xmax, ymin, ymax, zmin, zmax);
    } else {
        PAIRS_EXCEPTION("Domain partitioning type not implemented!\n");
    }

    dom_part->initialize(argc, argv);
}

void PairsSimulation::addArray(Array array) {
    int id = array.getId();
    auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array _a) { return _a.getId() == id; });
    PAIRS_ASSERT(a == std::end(arrays));
    arrays.push_back(array);
}

Array &PairsSimulation::getArray(array_t id) {
    auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array _a) { return _a.getId() == id; });
    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

Array &PairsSimulation::getArrayByName(std::string name) {
    auto a = std::find_if(arrays.begin(), arrays.end(), [name](Array _a) { return _a.getName() == name; });
    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

Array &PairsSimulation::getArrayByHostPointer(const void *h_ptr) {
    auto a = std::find_if(arrays.begin(), arrays.end(), [h_ptr](Array _a) { return _a.getHostPointer() == h_ptr; });
    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

void PairsSimulation::addProperty(Property prop) {
    int id = prop.getId();
    auto p = std::find_if(properties.begin(), properties.end(), [id](Property _p) { return _p.getId() == id; });
    PAIRS_ASSERT(p == std::end(properties));
    properties.push_back(prop);
}

Property &PairsSimulation::getProperty(property_t id) {
    auto p = std::find_if(properties.begin(), properties.end(), [id](Property _p) { return _p.getId() == id; });
    PAIRS_ASSERT(p != std::end(properties));
    return *p;
}

Property &PairsSimulation::getPropertyByName(std::string name) {
    auto p = std::find_if(properties.begin(), properties.end(), [name](Property _p) { return _p.getName() == name; });
    PAIRS_ASSERT(p != std::end(properties));
    return *p;
}

void PairsSimulation::addFeatureProperty(FeatureProperty feature_prop) {
    int id = feature_prop.getId();
    auto fp = std::find_if(feature_properties.begin(),
                           feature_properties.end(),
                           [id](FeatureProperty _fp) { return _fp.getId() == id; });
    PAIRS_ASSERT(fp == std::end(feature_properties));
    feature_properties.push_back(feature_prop);
}

FeatureProperty &PairsSimulation::getFeatureProperty(property_t id) {
    auto fp = std::find_if(feature_properties.begin(),
                           feature_properties.end(),
                           [id](FeatureProperty _fp) { return _fp.getId() == id; });
    PAIRS_ASSERT(fp != std::end(feature_properties));
    return *fp;
}

FeatureProperty &PairsSimulation::getFeaturePropertyByName(std::string name) {
    auto fp = std::find_if(feature_properties.begin(),
                           feature_properties.end(),
                           [name](FeatureProperty _fp) { return _fp.getName() == name; });
    PAIRS_ASSERT(fp != std::end(feature_properties));
    return *fp;
}

void PairsSimulation::copyArrayToDevice(Array &array) {
    int array_id = array.getId();
    if(!array_flags->isDeviceFlagSet(array_id)) {
        if(array.isStatic()) {
            PAIRS_DEBUG("Copying static array %s to device\n", array.getName().c_str());
            pairs::copy_static_symbol_to_device(array.getHostPointer(), array.getDevicePointer(), array.getSize());
        } else {
            PAIRS_DEBUG("Copying array %s to device\n", array.getName().c_str());
            pairs::copy_to_device(array.getHostPointer(), array.getDevicePointer(), array.getSize());
        }
    }
}

void PairsSimulation::copyArrayToHost(Array &array) {
    int array_id = array.getId();
    if(!array_flags->isHostFlagSet(array_id)) {
        if(array.isStatic()) {
            PAIRS_DEBUG("Copying static array %s to host\n", array.getName().c_str());
            pairs::copy_static_symbol_to_host(array.getDevicePointer(), array.getHostPointer(), array.getSize());
        } else {
            PAIRS_DEBUG("Copying array %s to host\n", array.getName().c_str());
            pairs::copy_to_host(array.getDevicePointer(), array.getHostPointer(), array.getSize());
        }
    }
}

void PairsSimulation::copyPropertyToDevice(Property &prop) {
    if(!prop_flags->isDeviceFlagSet(prop.getId())) {
        PAIRS_DEBUG("Copying property %s to device\n", prop.getName().c_str());
        pairs::copy_to_device(prop.getHostPointer(), prop.getDevicePointer(), prop.getTotalSize());
    }
}

void PairsSimulation::copyPropertyToHost(Property &prop) {
    if(!prop_flags->isHostFlagSet(prop.getId())) {
        PAIRS_DEBUG("Copying property %s to host\n", prop.getName().c_str());
        pairs::copy_to_host(prop.getDevicePointer(), prop.getHostPointer(), prop.getTotalSize());
    }
}

void PairsSimulation::copyFeaturePropertyToDevice(FeatureProperty &feature_prop) {
    PAIRS_DEBUG("Copying static array %s to device\n", feature_prop.getName().c_str());
    pairs::copy_static_symbol_to_device(feature_prop.getHostPointer(), feature_prop.getDevicePointer(), feature_prop.getArraySize());
}

void PairsSimulation::communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
    auto nsend_id = getArrayByHostPointer(send_sizes).getId();
    auto nrecv_id = getArrayByHostPointer(recv_sizes).getId();

    copyArrayToHost(nsend_id);
    array_flags->setHostFlag(nrecv_id);
    array_flags->clearDeviceFlag(nrecv_id);
    this->getDomainPartitioner()->communicateSizes(dim, send_sizes, recv_sizes);
    PAIRS_DEBUG("send_sizes=[%d, %d], recv_sizes=[%d, %d]\n", send_sizes[dim * 2 + 0], send_sizes[dim * 2 + 1], recv_sizes[dim * 2 + 0], recv_sizes[dim * 2 + 1]);
}

void PairsSimulation::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    auto send_buf_id = getArrayByHostPointer(send_buf).getId();
    auto recv_buf_id = getArrayByHostPointer(recv_buf).getId();
    auto send_offsets_id = getArrayByHostPointer(send_offsets).getId();
    auto recv_offsets_id = getArrayByHostPointer(recv_offsets).getId();
    auto nsend_id = getArrayByHostPointer(nsend).getId();
    auto nrecv_id = getArrayByHostPointer(nrecv).getId();

    copyArrayToHost(send_buf_id);
    copyArrayToHost(send_offsets_id);
    copyArrayToHost(recv_offsets_id);
    copyArrayToHost(nsend_id);
    copyArrayToHost(nrecv_id);
    array_flags->setHostFlag(recv_buf_id);
    array_flags->clearDeviceFlag(recv_buf_id);
    this->getDomainPartitioner()->communicateData(dim, elem_size, send_buf, send_offsets, nsend, recv_buf, recv_offsets, nrecv);

    /*
    // Debug messages
    const int elems_to_print = 5;

    // Send buffer debug
    for(int i = 0; i < 2; i++) {
        int nsnd = nsend[dim * 2 + i];

        PAIRS_DEBUG("send_buf=[");
        for(int j = 0; j < MIN(elems_to_print, nsnd); j++) {
            for(int k = 0; k < elem_size; k++) {
                PAIRS_DEBUG("%f,", send_buf[(send_offsets[dim * 2 + i] + j) * elem_size + k]);
            }
        }

        if(elems_to_print * 2 < nsnd) {
            PAIRS_DEBUG("\b ... ");
        }

        for(int j = MAX(elems_to_print, nsnd - elems_to_print); j < nsnd; j++) {
            for(int k = 0; k < elem_size; k++) {
                PAIRS_DEBUG("%f,", send_buf[(send_offsets[dim * 2 + i] + j) * elem_size + k]);
            }
        }

        PAIRS_DEBUG("\b]\n");
    }

    // Receive buffer debug
    for(int i = 0; i < 2; i++) {
        int nrec = nrecv[dim * 2 + i];

        PAIRS_DEBUG("recv_buf=[");
        for(int j = 0; j < MIN(elems_to_print, nrec); j++) {
            for(int k = 0; k < elem_size; k++) {
                PAIRS_DEBUG("%f,", recv_buf[(recv_offsets[dim * 2 + i] + j) * elem_size + k]);
            }
        }

        if(elems_to_print * 2 < nrec) {
            PAIRS_DEBUG("\b ... ");
        }

        for(int j = MAX(elems_to_print, nrec - elems_to_print); j < nrec; j++) {
            for(int k = 0; k < elem_size; k++) {
                PAIRS_DEBUG("%f,", recv_buf[(recv_offsets[dim * 2 + i] + j) * elem_size + k]);
            }
        }

        PAIRS_DEBUG("\b]\n");
    }
    */
}

void PairsSimulation::fillCommunicationArrays(int *neighbor_ranks, int *pbc, real_t *subdom) {
    this->getDomainPartitioner()->fillArrays(neighbor_ranks, pbc, subdom);
}

}
