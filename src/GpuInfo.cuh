#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <string>

#include <glm/vec3.hpp>
#include "./Mesh.h"

/*
* Move to seperate class
*/

struct MeshGpu {
    glm::vec3 * normalBuff;
    glm::vec3 * edgeBuff;
    glm::vec3 * faceBuff;
    Vertex * vertexBuffer;
    size_t faceSize;
    size_t vertexSize;
};

struct GpuInfo {
    MeshGpu * meshDev; //contains array of pointers that point to locations in infoBuffer
    size_t meshLen;
    void * infoBuffer; //contains all info.

    GpuInfo() = default; 

    __host__ GpuInfo(const std::vector<Mesh> & meshIn) {
        copyIntoDevice(meshIn);
    }

    private:

    void copyNormalBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyEdgeBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyFaceBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyVertexBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void setLength( const std::vector<Mesh> & meshIn, MeshGpu * meshHost);

    void copyIntoDevice(const std::vector<Mesh> & meshIn);

    size_t sumMeshSizes(const std::vector<Mesh> & meshIn) const;

    inline size_t sumMeshArr(const std::vector<Mesh> & meshIn) const {
        return sizeof(MeshGpu) * meshIn.size();
    }


    inline void handleCudaError(cudaError err) {
        if (err != cudaSuccess) {
            std::cout << err << "\n";
            throw std::runtime_error(" Issue with cuda; Error code : " + err);
        }
    }
};  

__global__ void printMeshInfo(GpuInfo inf);
