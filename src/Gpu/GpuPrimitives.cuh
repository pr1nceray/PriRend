#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <glm/vec3.hpp>
#include <iostream>
#include "../Primitives.cuh"

/*
* Used for parameter passing
*/
struct shaderInfo {
    __device__ void setRequired(const Ray * rayIn, const Ray * rayOut, const glm::vec3 * normal);
    glm::vec3 h;
    float ndotw_in;
    float ndotw_out;
    float hdotw_in;
    float hdotw_out;
    float ndotw_out_pow5;
    float ndotw_in_pow5;
};

struct MatGpu {
    __device__ explicit MatGpu() = default;
    __device__ glm::vec2 getIdx(const CollisionInfo * hitLoc) const;
    __device__ size_t getTextureIdx(TextInfo * inf, glm::vec2 * idx) const;
    __device__ float * getTextureColor(TextInfo * inf, glm::vec2 * idx) const;
    __device__ float * colorAt(const CollisionInfo * hitLoc, const shaderInfo * info) const;
    __device__ float baseDiffuse(glm::vec2 * idx, const shaderInfo * info) const;
    __device__ float baseSubsurface(glm::vec2 * idx, const shaderInfo * info) const;
    TextInfo * TextureArr[5];
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

inline void handleCudaError(cudaError err) {
    if (err != cudaSuccess) {
        std::cerr << err << "\n";
        cudaDeviceReset();
        throw std::runtime_error(" Issue with cuda; Error code : " + std::string(cudaGetErrorString(err)));
    }
}