#include "boundary_weights.hpp"

// Always include last generated interfaces
#include "last_generated.hpp"
namespace pairs {

void compute_boundary_weights(
    PairsRuntime *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    long unsigned int *comp_weight, long unsigned int *comm_weight) {

    const int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
    const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
    auto position_prop = ps->getPropertyByName("position");
    auto flags_prop = ps->getPropertyByName("flags");

    real_t *position_ptr = static_cast<real_t *>(position_prop.getHostPointer());
    int *flags_ptr = static_cast<int *>(flags_prop.getHostPointer());

    *comp_weight = 0;

    for(int i = 0; i < nlocal; i++) {
        if (pairs_host_interface::get_flags(flags_ptr, i) & (pairs::flags::INFINITE | pairs::flags::GLOBAL)) {
            continue;
        }

        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x >= xmin && pos_x < xmax &&
            pos_y >= ymin && pos_y < ymax &&
            pos_z >= zmin && pos_z < zmax) {
                (*comp_weight)++;
        }
    }

    // TODO: Count the number of ghosts that must be communicated with this block.
    // Note: The ghosts stored in this rank are NOT contained in the aabb of any of its blocks.
    //       And neighbor blocks are going to change after rebalancing.
    // const int nghost = ps->getTrackedVariableAsInteger("nghost");
    *comm_weight = 0;
}

void determine_non_empty_aabbs(PairsRuntime *ps, int num_aabbs, real_t *aabbs, int *non_empty_aabbs){
    const int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
    const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
    auto position_prop = ps->getPropertyByName("position");
    auto flags_prop = ps->getPropertyByName("flags");

    real_t *position_ptr = static_cast<real_t *>(position_prop.getHostPointer());
    int *flags_ptr = static_cast<int *>(flags_prop.getHostPointer());

    for(int i = 0; i < nlocal; ++i) {
        if (pairs_host_interface::get_flags(flags_ptr, i) & (pairs::flags::INFINITE | pairs::flags::GLOBAL)) {
            continue;
        }

        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);
        for(int n = 0; n < num_aabbs; ++n){
            if( pos_x >= aabbs[n*6 + 0] && pos_x < aabbs[n*6 + 1] &&
                pos_y >= aabbs[n*6 + 2] && pos_y < aabbs[n*6 + 3] &&
                pos_z >= aabbs[n*6 + 4] && pos_z < aabbs[n*6 + 5]) {
                    non_empty_aabbs[n] = true;
                    break;
            }
        }
    }
}

}
