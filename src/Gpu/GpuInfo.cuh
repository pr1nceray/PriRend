#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <glm/vec3.hpp>
#include "./GpuPrimitives.cuh"
#include "../Primitives.cuh"
#include "../Mesh.cuh"


struct GpuInfo {
    MeshGpu * meshDev; //contains array of pointers that point to locations in infoBuffer
    void * infoBuffer; //contains all info.
    size_t meshLen;
    MatGpu * matDev;
    size_t matLen;

    GpuInfo() = default; 


    __host__ GpuInfo(const std::vector<Mesh> & meshIn, const std::vector<Material> & matIn, const CameraInfo & camInf) {
        copyIntoDevice(meshIn, matIn);
        copyCameraData(camInf);
    }

    __host__ void freeResources();

    private:
    void copyIntoDevice(const std::vector<Mesh> & meshIn, const std::vector<Material> & matIn);
    void copyMeshData(const std::vector<Mesh> & meshIn);
    void copyMaterialData(const std::vector<Material> & matIn);
    void copyCameraData(const CameraInfo & camIn);

    
    /*
    * Internal functions For copying vertex data
    */
    template<typename T>
    void copyBuff(void * & start, const std::vector<T> * data, T * & write);
    void copyNormalBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyEdgeBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyFaceBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void copyVertexBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    void setLengthMesh( const std::vector<Mesh> & meshIn, MeshGpu * meshHost);
    size_t sumMeshSizes(const std::vector<Mesh> & meshIn) const; 
    size_t sumMeshArr(const std::vector<Mesh> & meshIn) const; 
    /*
    * Internal functions for copying material data
    */
    size_t sumMatArr(const std::vector<Material> & matIn) const;
    void copyMaterialIndex(const std::vector<Mesh> & meshIn, MeshGpu * meshHost);

};  

__global__ void printMeshInfo(GpuInfo inf);
__global__ void printMeshInfo(GpuInfo * inf);
__global__ void printMeshGlobal();
__global__ void printMaterialInfo();
__global__ void printBasicMaterialInfo();
__device__ extern GpuInfo * sceneInfo;
__device__ extern CameraInfo camInf;