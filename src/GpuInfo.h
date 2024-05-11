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

__device__ struct MeshGpu {
    glm::vec3 * normalBuff;
    glm::vec3 * edgeBuff;
    glm::vec2 * TQBuff;
};

struct GpuInfo {
    MeshGpu * meshDev; //contains array of pointers that point to locations in infoBuffer
    void * infoBuffer; //contains all info.

    __device__ GpuInfo() = default; 

    __host__ GpuInfo(const std::vector<Mesh> & meshIn) {
        size_t sizeOfMeshes = sumMeshSizes(meshIn); //size of the information
        size_t sizeOfArray = sumMeshArr(meshIn); // size of the struct to hold the information

        
        cudaError_t err = cudaMalloc((void **)(&meshDev), sizeOfArray); //malloc the array of ptrs
        handleCudaError(err);


        err = cudaMalloc((void **)(&infoBuffer), sizeOfMeshes); //malloc the info
        handleCudaError(err);

        MeshGpu * meshHost = new MeshGpu[sizeOfArray]; //create information holder on host


        /*
        * Copy over all the vertex buffers one after each other for cache purpouses.
        * Buffer copy is so that we dont lose infoBuffer.
        * take the size of the normalVector array, and copy over the data into where
        * BufferCopy is pointing. From there, set the pointer of mesh's to be accurate.
        * Then add the size that we copied.
        * same for TQ and edgemap.
        */
        void * bufferCpy = infoBuffer;
        for(size_t i = 0; i < meshIn.size(); ++i) { 
            size_t sizeOfNormal = sizeof(glm::vec3) * meshIn[i].FaceNormals.size();
            err = cudaMemcpy(&bufferCpy, meshIn[i].FaceNormals.data(), sizeOfNormal, cudaMemcpyHostToDevice);
            handleCudaError(err);
            meshHost[i].normalBuff = static_cast<glm::vec3 *>(bufferCpy);
            bufferCpy += sizeOfNormal;
        }

        for(size_t i = 0; i < meshIn.size(); ++i) { //copy over edgebuffers
            size_t sizeOfEdges = sizeof(glm::vec3) * meshIn[i].EdgeMap.size();
            err = cudaMemcpy(&bufferCpy, meshIn[i].FaceNormals.data(), sizeOfEdges, cudaMemcpyHostToDevice);
            handleCudaError(err);
            meshHost[i].edgeBuff = static_cast<glm::vec3 *>(bufferCpy);
            bufferCpy += sizeOfEdges;
        }
        
        for(size_t i = 0; i < meshIn.size(); ++i) { //copy over TQ buffers
            size_t sizeOfTQ = sizeof(glm::vec2) * meshIn[i].Indicies.size();
            err = cudaMemcpy(&bufferCpy, meshIn[i].FaceNormals.data(), sizeOfTQ, cudaMemcpyHostToDevice);
            handleCudaError(err);
            meshHost[i].edgeBuff = static_cast<glm::vec3 *>(bufferCpy);
            bufferCpy += sizeOfTQ;
        }

        //copy over all the info
        err = cudaMemcpy(meshDev, meshHost, sizeOfArray, cudaMemcpyHostToDevice);
        handleCudaError(err);
    }

    void copyIntoDevice();

    size_t sumMeshSizes(const std::vector<Mesh> & meshIn) const {
        size_t total = 0;
        for(size_t i = 0; i < meshIn.size(); ++i) {
            // face normals
            total += sizeof(glm::vec3) * meshIn[i].FaceNormals.size();
            // edgemap
            total += sizeof(glm::vec3) * meshIn[i].EdgeMap.size();
            // TQ buffer.
            total += sizeof(glm::vec2) * meshIn[i].Indicies.size(); 
        }
        return total;
    }

    size_t sumMeshArr(const std::vector<Mesh> & meshIn) const {
        return sizeof(GpuInfo) * meshIn.size();
    }


    void handleCudaError(cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error(" Issue with cuda; Error code : " + err);
        }
    }
};  
