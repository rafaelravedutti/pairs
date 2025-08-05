#include <blockforest/BlockForest.h>
#include <blockforest/BlockDataHandling.h>

#pragma once

namespace pairs {

class PairsRuntime;

void relocate_particle(PairsRuntime *ps, int dst, int src){
    for(auto &prop: ps->getProperties()) {
        if(!prop.isVolatile()) {
            auto prop_type = prop.getType();

            if(prop_type == pairs::Prop_Vector) {
                auto vector_ptr = ps->getAsVectorProperty(prop);
                constexpr int nelems = 3;

                for(int e = 0; e < nelems; e++) {
                    vector_ptr(dst, e) = vector_ptr(src, e);
                }
            } else if(prop_type == pairs::Prop_Matrix) {
                auto matrix_ptr = ps->getAsMatrixProperty(prop);
                constexpr int nelems = 9;

                for(int e = 0; e < nelems; e++) {
                    matrix_ptr(dst, e) = matrix_ptr(src, e);
                }
            } else if(prop_type == pairs::Prop_Quaternion) {
                auto quat_ptr = ps->getAsQuaternionProperty(prop);
                constexpr int nelems = 4;

                for(int e = 0; e < nelems; e++) {
                    quat_ptr(dst, e) = quat_ptr(src, e);
                }
            } else if(prop_type == pairs::Prop_Integer) {
                auto int_ptr = ps->getAsIntegerProperty(prop);
                int_ptr(dst) = int_ptr(src);
            } else if(prop_type == pairs::Prop_UInt64) {
                auto uint64_ptr = ps->getAsUInt64Property(prop);
                uint64_ptr(dst) = uint64_ptr(src);
            } else if(prop_type == pairs::Prop_Real) {
                auto float_ptr = ps->getAsFloatProperty(prop);
                float_ptr(dst) = float_ptr(src);
            } else {
                std::cerr << "relocate_particle(): Invalid property type!" << std::endl;
                return;
            }
        }
    }
}

}

namespace walberla {

namespace internal {

class ParticleDeleter {
    friend bool operator==(const ParticleDeleter& lhs, const ParticleDeleter& rhs);

public:
    ParticleDeleter(pairs::PairsRuntime *ps_, const math::AABB& aabb_) : ps(ps_), aabb(aabb_) {}

    ~ParticleDeleter() {
        int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        auto position = ps->getAsVectorProperty(ps->getPropertyByName("position"));
        auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));

        int ndeleted = 0;
        int *goneIdx = new int[nlocal];
        
        for (int i=0; i<nlocal; ++i) {
            if (flags(i) & (pairs::flags::INFINITE | pairs::flags::GLOBAL))  continue;

            const real_t pos_x = position(i, 0);
            const real_t pos_y = position(i, 1);
            const real_t pos_z = position(i, 2);

            if( aabb.contains(pos_x, pos_y, pos_z)) {
                goneIdx[ndeleted] = i;
                ++ndeleted;
            }
        }

        int beg = 0;
        int end = ndeleted - 1;
        int i = nlocal - 1;
        while ((i > goneIdx[beg]) && (beg <= end)) {
            if(i == goneIdx[end]){
                --end;
            }
            else{
                pairs::relocate_particle(ps, goneIdx[beg], i);
                ++beg;
            }
            --i;
        }
        
        delete[] goneIdx;
        
        ps->setTrackedVariableAsInteger("nlocal", nlocal - ndeleted);
        ps->setTrackedVariableAsInteger("nghost", 0);
    }

private:
    pairs::PairsRuntime *ps;
    math::AABB aabb;
};

inline bool operator==(const ParticleDeleter& lhs, const ParticleDeleter& rhs) {
    return lhs.aabb == rhs.aabb;
}

} // namespace internal

class ParticleDataHandling : public blockforest::BlockDataHandling<internal::ParticleDeleter> {
private:
    pairs::PairsRuntime *ps;

public:
    ParticleDataHandling(pairs::PairsRuntime *ps_) : ps(ps_) {}
    ~ParticleDataHandling() override = default;

    internal::ParticleDeleter *initialize(IBlock *const block) override {
        return new internal::ParticleDeleter(ps, block->getAABB());
    }

    void serialize(IBlock *const block, const BlockDataID& id, mpi::SendBuffer& buffer) override {
        serializeImpl(static_cast<Block*>(block), id, buffer, 0, false);
    }

    internal::ParticleDeleter* deserialize(IBlock *const block) override {
        return initialize(block);
    }

    void deserialize(IBlock *const block, const BlockDataID& id, mpi::RecvBuffer& buffer) override {
        deserializeImpl(block, id, buffer);
    }

    void serializeCoarseToFine(Block *const block, const BlockDataID& id, mpi::SendBuffer& buffer, const uint_t child) override {
        serializeImpl(block, id, buffer, child, true);
    }

    void serializeFineToCoarse(Block *const block, const BlockDataID& id, mpi::SendBuffer& buffer) override {
        serializeImpl(block, id, buffer, 0, false);
    }

    internal::ParticleDeleter *deserializeCoarseToFine(Block *const block) override {
        return initialize(block);
    }

    internal::ParticleDeleter *deserializeFineToCoarse(Block *const block) override {
        return initialize(block);
    }

    void deserializeCoarseToFine(Block *const block, const BlockDataID& id, mpi::RecvBuffer& buffer) override {
        deserializeImpl(block, id, buffer);
    }

    void deserializeFineToCoarse(Block *const block, const BlockDataID& id, mpi::RecvBuffer& buffer, const uint_t) override {
        deserializeImpl(block, id, buffer);
    }

    void serializeImpl(Block *const block, const BlockDataID&, mpi::SendBuffer& buffer, const uint_t child, bool check_child) {
        auto ptr = buffer.allocate<int>();
        double aabb_check[6];

        if(check_child) {
            const auto child_id = BlockID(block->getId(), child);
            const auto child_aabb = block->getForest().getAABBFromBlockId(child_id);
            aabb_check[0] = child_aabb.xMin();
            aabb_check[1] = child_aabb.xMax();
            aabb_check[2] = child_aabb.yMin();
            aabb_check[3] = child_aabb.yMax();
            aabb_check[4] = child_aabb.zMin();
            aabb_check[5] = child_aabb.zMax();
        } else {
            const auto aabb = block->getAABB();
            aabb_check[0] = aabb.xMin();
            aabb_check[1] = aabb.xMax();
            aabb_check[2] = aabb.yMin();
            aabb_check[3] = aabb.yMax();
            aabb_check[4] = aabb.zMin();
            aabb_check[5] = aabb.zMax();
        }

        int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        auto position = ps->getAsVectorProperty(ps->getPropertyByName("position"));
        auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
        int nserialized = 0;
        int *goneIdx = new int[nlocal];

        for (int i=0; i<nlocal; ++i) {
            if (flags(i) & (pairs::flags::INFINITE | pairs::flags::GLOBAL)) continue;
            const real_t pos_x = position(i, 0);
            const real_t pos_y = position(i, 1);
            const real_t pos_z = position(i, 2);

            // Important: When rebalancing, it is assumed that all particles are within domain bounds.  
            // If a particle's center of mass lies outside the domain, it won't be contained
            // in any of the checked blocks during serialization. In that case, the particle  
            // can become disassociated from its owner if the new block it should belong to is  
            // not an immediate neighbor to its owner rank. (if it's in an immediate neighbor, it will be exchanged)
            if( pos_x >= aabb_check[0] && pos_x < aabb_check[1] &&
                pos_y >= aabb_check[2] && pos_y < aabb_check[3] &&
                pos_z >= aabb_check[4] && pos_z < aabb_check[5]) {

                goneIdx[nserialized] = i;
                ++nserialized;
                
                for(auto &prop: ps->getProperties()) {
                    if(!prop.isVolatile()) {
                        auto prop_type = prop.getType();

                        if(prop_type == pairs::Prop_Vector) {
                            auto vector_ptr = ps->getAsVectorProperty(prop);
                            constexpr int nelems = 3;

                            for(int e = 0; e < nelems; e++) {
                                buffer << vector_ptr(i, e);
                            }
                        } else if(prop_type == pairs::Prop_Matrix) {
                            auto matrix_ptr = ps->getAsMatrixProperty(prop);
                            constexpr int nelems = 9;

                            for(int e = 0; e < nelems; e++) {
                                buffer << matrix_ptr(i, e);
                            }
                        } else if(prop_type == pairs::Prop_Quaternion) {
                            auto quat_ptr = ps->getAsQuaternionProperty(prop);
                            constexpr int nelems = 4;

                            for(int e = 0; e < nelems; e++) {
                                buffer << quat_ptr(i, e);
                            }
                        } else if(prop_type == pairs::Prop_Integer) {
                            auto int_ptr = ps->getAsIntegerProperty(prop);
                                buffer << int_ptr(i);
                        } else if(prop_type == pairs::Prop_UInt64) {
                            auto uint64_ptr = ps->getAsUInt64Property(prop);
                                buffer << uint64_ptr(i);
                        } else if(prop_type == pairs::Prop_Real) {
                            auto float_ptr = ps->getAsFloatProperty(prop);
                                buffer << float_ptr(i);
                        } else {
                            std::cerr << "serializeImpl(): Invalid property type!" << std::endl;
                            return;
                        }
                    }
                }
                // TODO: serialize contact history data as well
            }
        }

        // Here we replace serialized particles with the remaining locals 
        // (Traverse locals in reverse order and move them to empty slots)
        // Ghosts are ignored since they become invalid after rebalancing
        int beg = 0;
        int end = nserialized - 1;
        int i = nlocal - 1;
        while ((i > goneIdx[beg]) && (beg <= end)) {
            if(i == goneIdx[end]){
                --end;
            }
            else{
                pairs::relocate_particle(ps, goneIdx[beg], i);
                ++beg;
            }
            --i;
        }

        delete[] goneIdx;

        ps->setTrackedVariableAsInteger("nlocal", nlocal - nserialized);
        ps->setTrackedVariableAsInteger("nghost", 0);
        
        *ptr = (int) nserialized;
    }

    void deserializeImpl(IBlock *const, const BlockDataID&, mpi::RecvBuffer& buffer) {
        int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        real_t real_tmp;
        int int_tmp;
        int nrecv;
        uint64_t uint64_tmp;

        buffer >> nrecv;
        
        // TODO: Check if there is enough particle capacity for the new particles, when there is not,
        // all properties and arrays which have particle_capacity as one of their dimensions must be reallocated
        // int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
        // PAIRS_ASSERT(nlocal + nrecv < particle_capacity);

        for(int i = 0; i < nrecv; ++i) {
            for(auto &prop: ps->getProperties()) {
                if(!prop.isVolatile()) {
                    auto prop_type = prop.getType();

                    if(prop_type == pairs::Prop_Vector) {
                        auto vector_ptr = ps->getAsVectorProperty(prop);
                        constexpr int nelems = 3;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            vector_ptr(nlocal + i, e) = real_tmp;
                        }
                    } else if(prop_type == pairs::Prop_Matrix) {
                        auto matrix_ptr = ps->getAsMatrixProperty(prop);
                        constexpr int nelems = 9;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            matrix_ptr(nlocal + i, e) = real_tmp;
                        }
                    } else if(prop_type == pairs::Prop_Quaternion) {
                        auto quat_ptr = ps->getAsQuaternionProperty(prop);
                        constexpr int nelems = 4;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            quat_ptr(nlocal + i, e) = real_tmp;
                        }
                     } else if(prop_type == pairs::Prop_Integer) {
                        auto int_ptr = ps->getAsIntegerProperty(prop);
                        buffer >> int_tmp;
                        int_ptr(nlocal + i) = int_tmp;
                    } else if(prop_type == pairs::Prop_UInt64) {
                        auto uint64_ptr = ps->getAsUInt64Property(prop);
                        buffer >> uint64_tmp;
                        uint64_ptr(nlocal + i) = uint64_tmp;
                    } else if(prop_type == pairs::Prop_Real) {
                        auto float_ptr = ps->getAsFloatProperty(prop);
                        buffer >> real_tmp;
                        float_ptr(nlocal + i) = real_tmp;
                    } else {
                        std::cerr << "deserializeImpl(): Invalid property type!" << std::endl;
                        return;
                    }
                }
            }
        }
        
        ps->setTrackedVariableAsInteger("nlocal", nlocal + nrecv);
        ps->setTrackedVariableAsInteger("nghost", 0);
    }
};

} // namespace walberla
