#include <blockforest/BlockForest.h>
#include <blockforest/BlockDataHandling.h>

namespace pairs {

class PairsRuntime;

}

namespace walberla {

namespace internal {

class ParticleDeleter {
    friend bool operator==(const ParticleDeleter& lhs, const ParticleDeleter& rhs);

public:
    ParticleDeleter(const math::AABB& aabb) : aabb_(aabb) {}
    ~ParticleDeleter() {}

private:
    math::AABB aabb_;
};

inline bool operator==(const ParticleDeleter& lhs, const ParticleDeleter& rhs) {
    return lhs.aabb_ == rhs.aabb_;
}

} // namespace internal

class ParticleDataHandling : public blockforest::BlockDataHandling<internal::ParticleDeleter> {
private:
    pairs::PairsRuntime *ps;

public:
    ParticleDataHandling(pairs::PairsRuntime *ps_) : ps(ps_) {}
    ~ParticleDataHandling() override = default;

    internal::ParticleDeleter *initialize(IBlock *const block) override {
        return new internal::ParticleDeleter(block->getAABB());
    }

    void serialize(IBlock *const block, const BlockDataID& id, mpi::SendBuffer& buffer) override {
        serializeImpl(static_cast<Block *const>(block), id, buffer, 0, false);
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
        auto ptr = buffer.allocate<uint_t>();
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

        for(auto& prop: ps->getProperties()) {
            if(!prop.isVolatile()) {
                ps->copyPropertyToHost(prop, WriteAfterRead);
            }
        }

        auto position = ps->getAsVectorProperty(ps->getPropertyByName("position"));
        int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        int i = 0;
        int nserialized = 0;

        while(i < nlocal) {
            const real_t pos_x = position(i, 0);
            const real_t pos_y = position(i, 1);
            const real_t pos_z = position(i, 2);

            if( pos_x > aabb_check[0] && pos_x <= aabb_check[1] &&
                pos_y > aabb_check[2] && pos_y <= aabb_check[3] &&
                pos_z > aabb_check[4] && pos_z <= aabb_check[5]) {

                nlocal--;

                for(auto &prop: ps->getProperties()) {
                    if(!prop.isVolatile()) {
                        auto prop_type = prop.getType();

                        if(prop_type == Prop_Vector) {
                            auto vector_ptr = ps->getAsVectorProperty(prop);
                            constexpr int nelems = 3;

                            for(int e = 0; e < nelems; e++) {
                                buffer << vector_ptr(i, e);
                                vector_ptr(i, e) = vector_ptr(nlocal, e);
                            }
                        } else if(prop_type == Prop_Matrix) {
                            auto matrix_ptr = ps->getAsMatrixProperty(prop);
                            constexpr int nelems = 9;

                            for(int e = 0; e < nelems; e++) {
                                buffer << matrix_ptr(i, e);
                                matrix_ptr(i, e) = matrix_ptr(nlocal, e);
                            }
                        } else if(prop_type == Prop_Quaternion) {
                            auto quat_ptr = ps->getAsQuaternionProperty(prop);
                            constexpr int nelems = 4;

                            for(int e = 0; e < nelems; e++) {
                                buffer << quat_ptr(i, e);
                                quat_ptr(i, e) = quat_ptr(nlocal, e);
                            }
                        } else if(prop_type == Prop_Integer) {
                            auto int_ptr = ps->getAsIntegerProperty(prop);
                            buffer << int_ptr(i);
                            int_ptr(i) = int_ptr(nlocal);
                        } else if(prop_type == Prop_Real) {
                            auto float_ptr = ps->getAsFloatProperty(prop);
                            buffer << float_ptr(i);
                            float_ptr(i) = float_ptr(nlocal);
                        } else {
                            std::cerr << "serializeImpl(): Invalid property type!" << std::endl;
                            return;
                        }
                    }
                }
            }

            // TODO: serialize contact history data as well
            nserialized++;
        }

        ps->setTrackedVariableAsInteger("nlocal", nlocal);
        *ptr = (uint_t) nserialized;
    }

    void deserializeImpl(IBlock *const, const BlockDataID&, mpi::RecvBuffer& buffer) {
        int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
        real_t real_tmp;
        int int_tmp;
        uint_t nrecv;

        buffer >> nrecv;

        // TODO: Check if there is enough particle capacity for the new particles, when there is not,
        // all properties and arrays which have particle_capacity as one of their dimensions must be reallocated
        PAIRS_ASSERT(nlocal + nrecv < particle_capacity);

        for(int i = 0; i < nrecv; ++i) {
            for(auto &prop: ps->getProperties()) {
                if(!prop.isVolatile()) {
                    auto prop_type = prop.getType();

                    if(prop_type == Prop_Vector) {
                        auto vector_ptr = ps->getAsVectorProperty(prop);
                        constexpr int nelems = 3;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            vector_ptr(nlocal + i, e) = real_tmp;
                        }
                    } else if(prop_type == Prop_Matrix) {
                        auto matrix_ptr = ps->getAsMatrixProperty(prop);
                        constexpr int nelems = 9;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            matrix_ptr(nlocal + i, e) = real_tmp;
                        }
                    } else if(prop_type == Prop_Quaternion) {
                        auto quat_ptr = ps->getAsQuaternionProperty(prop);
                        constexpr int nelems = 4;

                        for(int e = 0; e < nelems; e++) {
                            buffer >> real_tmp;
                            quat_ptr(nlocal + i, e) = real_tmp;
                        }
                     } else if(prop_type == Prop_Integer) {
                        auto int_ptr = ps->getAsIntegerProperty(prop);
                        buffer >> int_tmp;
                        int_ptr(nlocal + i) = int_tmp;
                    } else if(prop_type == Prop_Real) {
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
    }
};

} // namespace walberla
