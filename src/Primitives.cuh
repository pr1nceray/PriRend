#pragma once
#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <limits>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.cuh"

const int WIDTH = 1152;
const int HEIGHT = 648;
const int CHANNEL = 4;


const int SPP = 256;
const int BOUNCES = 8;

const float FOV_X = 90;
const float FOV_Y = 60;

const float ASPECT_RATIO = static_cast<float>(WIDTH)/HEIGHT;
const float DESIRED_AR = 16.0f/9.0f;
const float epsil = .000001;
const float randEpsil = .0000000001f;

const float pi = 3.1415926535;

const int TEXTURENUM = 7;

struct Vertex {
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;
};

struct CameraInfo {
    glm::vec3 center;
    glm::vec3 lookingDir;
    float zoom;
};

struct TextInfo {
    TextInfo() : basic(true) {
    }
    TextInfo(cudaTextureObject_t obj) : 
    text(obj), basic(false){
    }
    cudaTextureObject_t text;
    bool basic;
    float4 basicColor;
};

struct CollisionInfo {
    glm::vec2 CollisionPoint;
    Vertex * A;
    Vertex * B;
    Vertex * C;
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
__device__ const bool isZero(const glm::vec3 * in);
/*
* Generate uniformly random float on the range -.5, .5
*/
__host__ float generateRandomFloatH();
__device__ float generateRandomFloatD(curandState * state);
__device__ float generateNormalFloatD(curandState * state);
__device__ float generateInvNormalFloatD(curandState * const state);
/*
* Generate uniformly random float on the range -1, 1
*/

__host__ uint8_t generateRandomNumH();

/*
* Generates a random normalized vector
*/
__host__ glm::vec3 generateRandomVecH();
__device__ glm::vec3 generateRandomVecD(curandState * state);
__device__ glm::vec3 generateNormalVecD(curandState * state);
__device__ glm::vec3 generateInvNormalVecD(curandState * state);