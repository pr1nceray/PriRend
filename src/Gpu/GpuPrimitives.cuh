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
    __device__ glm::vec3 samplePoint(const CollisionInfo * hitloc, const curandState_t * state) const;
    __device__ glm::vec2 getIdx(const CollisionInfo * hitLoc) const;
    __device__ const float4 getTextureColor(const TextInfo * inf, const glm::vec2 * idx) const;
    __device__ const float4 colorAt(const CollisionInfo * hitLoc, const shaderInfo * info) const;
    __device__ float baseDiffuse(const glm::vec2 * idx, const shaderInfo * info) const;
    __device__ float baseSubsurface(const glm::vec2 * idx, const shaderInfo * info) const;
    __device__ float baseMetallic(const glm::vec2 * idx, const shaderInfo * info) const;
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
    bool isSmooth;

    __device__ glm::vec3 generateRandomVecOnFace(const CollisionInfo * info, curandState * state) const;
    __device__ glm::vec3 generateReflectiveVecOnFace(const CollisionInfo * info, const glm::vec3 & dir) const;
    __device__ glm::vec3 generateRefractiveVecOnFace(const CollisionInfo * info, const glm::vec3 & dir) const;
    __device__ glm::vec3 generateRoughVecOnFace(const CollisionInfo * info, const glm::vec3 & dir, curandState * state) const;

    __device__ MatGpu const & getMaterial() const;
    __device__ glm::vec3 getFaceNormal(const CollisionInfo * inf) const;
};

__device__ void printTextures(TextInfo * text);
__device__ glm::vec3 getBays(const CollisionInfo * inf);
inline void handleCudaError(cudaError err) {
    if (err != cudaSuccess) {
        std::cerr << err << "\n";
        cudaDeviceReset();
        throw std::runtime_error(" Issue with cuda; Error : " + std::string(cudaGetErrorString(err)) + "\n");
    }
}