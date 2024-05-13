#pragma once

#include <utility>
#include <math.h>
#include <limits.h>
#include <curand.h>
#include <curand_kernel.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "./Color.cuh"
#include "./Object.cuh"
#include "./Primitives.cuh"
#include "./GpuInfo.cuh"

const int WIDTH = 1280;
const int HEIGHT = 960;
const int CHANNEL = 3;

const int FOV_Y = 60;
const int FOV_X = 90;

const int SPP = 512;
const int BOUNCES = 32;

const float ASPECT_RATIO = static_cast<float>(WIDTH)/HEIGHT;
const float epsil = .000001;


using evalInfo = std::pair<bool, CollisionInfo>;


/*
* Muller Trom method
*/
__device__ bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & Edge1, const glm::vec3 & Edge2, CollisionInfo * out);

/*
* Checks for mesh collisions
*/
__device__ bool intersectsMesh(const Mesh & mesh, const Ray & ray, CollisionInfo * closestFace);
/*
* Responsible for checking if a ray collides with an object for all objects in the scene. 
*/
__device__ CollisionInfo checkCollisions(const Ray & ray, const GpuInfo * info);


/*
* Itertative version of eval bcuz cuda sucks at recursion.
*/
__device__ Color evalIter(Ray & ray, const GpuInfo * info, curandState * const randState, const int bounceCount);

/*
* eval is the recursive portion that checks for collisions, and applies bsdf
*/
__device__ Color eval(Ray & ray, const GpuInfo * info, curandState * const randState, const int bounceCount);

/*
* TraceRay is responsible for creating the ray and giving it a direction based on u,v.
* Takes in the objects to determine collisions.
*/
__device__ Color traceRay(float u, float v, curandState * const randState, GpuInfo * info);

/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
__global__ void spawnRay(GpuInfo info, int seed, uint8_t * colorArr);

/*
* Achieves the exact same as spawnRay, but does so progressively so that we can write to the image
* Slightly slower due to the need to need to run to cpu every time we want to update the image.
*/
__global__ void spawnRayProgressive(GpuInfo info, int seed, float * colorArr);


/*
* Convert a float to a color
*/
__device__ void convertSingle(const float * num, uint8_t *out);

/*
* Convert the float array to a uint8_t array
*/
__global__ void convertArr(const float * colorArr, uint8_t * out);

/*
* Clear array in preparation for writing
*/
__global__ void wipeArr(float * colorArr);
