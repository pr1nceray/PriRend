#pragma once
#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <limits>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.h"

struct Vertex {
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;
};

struct CollisionInfo {
    glm::vec2 CollisionPoint;
    float distanceMin;
    int faceIdx;
    int meshIdx;

    __host__ __device__ CollisionInfo() :
    CollisionPoint(glm::vec2(0, 0)),
    distanceMin(__FLT_MAX__),
    faceIdx(-1), meshIdx(-1) {
    }
};

struct Ray {
    glm::vec3 Origin;
    glm::vec3 Dir;

    __host__ __device__ Ray () {
    }
    
    __host__ __device__ Ray(const glm::vec3 & originIn, const glm::vec3 dirIn)  :
    Origin(originIn), Dir(dirIn) {
    }
};

__host__ __device__ void normalizeRayDir(Ray & ray);

/*
* Generate uniformly random float on the range -.5, .5
*/
__host__ float generateRandomFloatH() {
    return (static_cast<float>(std::rand() - RAND_MAX/2)/RAND_MAX);
}
__device__ float generateRandomFloatD(curandStatus * state) {
}

/*
* Generate uniformly random float on the range -1, 1
*/

__host__ uint8_t generateRandomNumH() {
    return static_cast<uint8_t>(rand() % 255);
}
__device__ uint8_t generateRandomNumD() {
    return static_cast<uint8_t>(rand() % 255);
}
/*
* Generates a random normalized vector
*/
__host__ glm::vec3 generateRandomVecH() {
    return glm::normalize(glm::vec3(
        generateRandomFloatH(), generateRandomFloatH(), generateRandomFloatH()));
}

__device__ glm::vec3 generateRandomVecD() {
    return glm::normalize(glm::vec3(
        generateRandomFloatD(), generateRandomFloatD(), generateRandomFloatD()));
}
