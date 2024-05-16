#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <glm/vec3.hpp>
#include "../Primitives.cuh"



struct MatGpu{
    
    __device__ explicit MatGpu() = default;
    TextInfo * diffuse;
   __device__ float *diffuseAtPoint(const CollisionInfo * hitLoc) const  ;
};

struct MeshGpu {
    glm::vec3 * normalBuff;
    glm::vec3 * edgeBuff;
    glm::ivec3 * faceBuff;
    Vertex * vertexBuffer;
    size_t faceSize;
    size_t vertexSize;
    size_t matIdx;

    __device__ Ray generateRandomVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const;
    __device__ Ray generateLambertianVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const;
    __device__ Ray generateReflectiveVecOnFace(const size_t faceIdx, const glm::vec3 & dir, const glm::vec3 & origin) const;

    __device__ MatGpu const & getMaterial() const;
    __device__ const glm::vec3 & getFaceNormal(size_t idx) const;
};

__device__ void printTextures(TextInfo * text);