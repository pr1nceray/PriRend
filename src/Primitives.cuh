#pragma once
#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>


#include <limits>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.cuh"

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
__host__ float generateRandomFloatH();
__device__ float generateRandomFloatD(curandState * state);

/*
* Generate uniformly random float on the range -1, 1
*/

__host__ uint8_t generateRandomNumH();

/*
* Generates a random normalized vector
*/
__host__ glm::vec3 generateRandomVecH();
__device__ glm::vec3 generateRandomVecD(curandState * state);