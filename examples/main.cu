#include <iostream>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* func) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", func, cudaGetErrorString(err));
        exit(err);
    }
}

template< typename Type >
class PairsVector3 {
public:
    PairsVector3() = default;

    // If the constructor is called from device, v_ is automatically allocated on 
    // device because it's a static array embeded in the object itself 
    __host__ __device__ PairsVector3( Type x, Type y, Type z ) {
        v_[0] = x;
        v_[1] = y;
        v_[2] = z;
    }

    __host__ __device__ Type& operator[]( int index ) { 
        return v_[index]; 
    }
    __host__ __device__ const Type& operator[] ( int index ) const { 
        return v_[index]; 
    }

private:
    Type v_[3] = {Type(), Type(), Type()};
};


struct PairsObjects {
    double *position_h;
    double *position_d;
};


class PairsAccessor{
    private:
    PairsObjects *pobj_h;
    PairsObjects *pobj_d;

    public:
    // PairsAccessor is only constructable from host, but its getters and setters can be called from both host and device
    __host__ PairsAccessor(PairsObjects *pobj_): pobj_h(pobj_) {

        // NOTE: Here we copy pobj_h to device, but we will use pobj_d to ONLY work with device pointers 
        // So for example, we only try to access pobj_d->position_d, NOT pobj_d->position_h 
        // NOTE: What's copied to device here is the PairsObject, which is a bunch of pointers (including valid device pointers like position_d)
        // TODO (maybe): Split PairsObjects into two structs, one holding host pointers, the other holding device pointers.
        cudaMalloc(&pobj_d, sizeof(PairsObjects));
        cudaMemcpy(pobj_d, pobj_h, sizeof(PairsObjects), cudaMemcpyHostToDevice);
    }

    __host__ __device__ int getTest(){
        #ifdef __CUDA_ARCH__ 
            return 12;
        #else
            return 34;
        #endif
    }

    // If this function is called from device, it returns a PairsVector3 that's constructed on device
    // If this function is called from host, it returns a PairsVector3 that's constructed on host
    __host__ __device__ PairsVector3<double> getPosition(const size_t i) const {
        #ifdef __CUDA_ARCH__ 
            // Assume postion_d points to uptodate data (we can't do copyPropertyToDevice from __device__ )
            return PairsVector3<double>(pobj_d->position_d[i*3 + 0], pobj_d->position_d[i*3 + 1], pobj_d->position_d[i*3 + 2]);
        #else
            // Here we can do copyPropertyToHost (ReadOnly) if needed, to make sure position_h has uptodate data
            return PairsVector3<double>(pobj_h->position_h[i*3 + 0], pobj_h->position_h[i*3 + 1], pobj_h->position_h[i*3 + 2]);
        #endif 
            
    }

    __host__ __device__ void setPosition(const size_t i, PairsVector3<double> const &vec) {
        #ifdef __CUDA_ARCH__ 
            // Assume vec is on device
            pobj_d->position_d[i*3 + 0] = vec[0]; 
            pobj_d->position_d[i*3 + 1] = vec[1]; 
            pobj_d->position_d[i*3 + 2] = vec[2];
            // Assume we don't need position_h data back on host (we can't do copyPropertyToHost from __device__)
        #else  
            // Assume vec is on host
            pobj_h->position_h[i*3 + 0] = vec[0]; 
            pobj_h->position_h[i*3 + 1] = vec[1]; 
            pobj_h->position_h[i*3 + 2] = vec[2];
            // Here we can do copyPropertyToDevice (WriteOnly) if needed (just so host and device data match)
        #endif 
    
    }
    
};


__global__ void mykernel(PairsAccessor ac){
    printf("getTest from device = %d\n", ac.getTest());

    PairsVector3<double> pos(7,8,9); 
    ac.setPosition(0, pos);
    printf("getPosition(0) from device = (%f, %f, %f) \n", ac.getPosition(0)[0], ac.getPosition(0)[1], ac.getPosition(0)[2]);
}


int main(int argc, char **argv) {

    PairsObjects *pobj = new PairsObjects;

    // User doesn't bother with the stuff below, they are done when PairsSimulation is initialized
    //----------------------------------------------------------------------------------------------------
    int numParticles = 1;
    int numElements = numParticles * 3;
    pobj->position_h = new double[numElements];
    cudaMalloc(&pobj->position_d, numElements * sizeof(double));
    cudaMemcpy(pobj->position_d, pobj->position_h, numElements * sizeof(double), cudaMemcpyHostToDevice);
    //----------------------------------------------------------------------------------------------------

    PairsAccessor ac(pobj);
    printf("getTest from host = %d\n", ac.getTest());

    PairsVector3<double> pos(1.2, 3.4, 5.6); 
    ac.setPosition(0, pos);
    printf("getPosition(0) from host = (%f, %f, %f) \n", ac.getPosition(0)[0], ac.getPosition(0)[1], ac.getPosition(0)[2]);
    

    mykernel<<<1,1>>>(ac);
    checkCudaError(cudaDeviceSynchronize(), "mykernel");

    // In mykernel, we modify the position of the particle ON DEVICE
    // TODO: To reflect this modification on the host, we need a 'sync' function along with getters and setters (eg: 'syncPosition')
    // to make sure both host and device data are uptodate and in sync with eachother.
    // But unlike getters and setters, sync functions are only callable form host
    // The 'sync' function copies the property to device if its host flag is set, or copies property to host if its device flag is set
    // If setter is called from host, set host flag and unset device flag for that property
    // If setter is called from device, set device flag and unset host flag for that property

}


