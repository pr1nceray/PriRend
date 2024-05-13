#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include <vector>
#include <string>

#include <glm/vec3.hpp>
#include "./Mesh.cuh"

/*
* Move to seperate class
*/

__device__ GpuInfo sceneInfo;

struct MeshGpu {
    glm::vec3 * normalBuff;
    glm::vec3 * edgeBuff;
    glm::ivec3 * faceBuff;
    Vertex * vertexBuffer;
    size_t faceSize;
    size_t vertexSize;

    __device__ Ray generateRandomVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const;
    __device__ Ray generateLambertianVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const;
    __device__ Ray generateReflectiveVecOnFace(const size_t faceIdx, const glm::vec3 & dir, const glm::vec3 & origin) const;

    __device__ Material const & getMaterial() const;
    __device__ const glm::vec3 & getFaceNormal(size_t idx) const;
};


struct GpuInfo {
    MeshGpu * meshDev; //contains array of pointers that point to locations in infoBuffer
    size_t meshLen;
    void * infoBuffer; //contains all info.

    GpuInfo() = default; 


    __host__ GpuInfo(const std::vector<Mesh> & meshIn) {
        copyIntoDevice(meshIn);
    }

    __host__ void freeResources();
    private:


    template<typename T>
    void copyBuff(void * & start, const std::vector<T> * data, T * & write);
    void copyNormalBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyEdgeBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyFaceBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyVertexBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void setLength( const std::vector<Mesh> & meshIn, MeshGpu * meshHost);

    void copyIntoDevice(const std::vector<Mesh> & meshIn);
    size_t sumMeshSizes(const std::vector<Mesh> & meshIn) const;
    size_t sumMeshArr(const std::vector<Mesh> & meshIn) const; 


};  

inline void handleCudaError(cudaError err) {
    if (err != cudaSuccess) {
        std::cerr << err << "\n";
        cudaDeviceReset();
        throw std::runtime_error(" Issue with cuda; Error code : " + std::string(cudaGetErrorString(err)));
    }
}
__global__ void printMeshInfo(GpuInfo inf);
