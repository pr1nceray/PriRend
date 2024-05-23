#include "Primitives.cuh"

 __host__ __device__ void normalizeRayDir(Ray & ray) {
    ray.Dir = glm::normalize(ray.Dir);
}

/*
* Flip a ray along the normal for PURE reflective materials. 
*/
glm::vec3 flipRayNormal(const Ray & ray, const glm::vec3 & normal) {
    return ray.Dir - (2.0f * (glm::dot(ray.Dir, normal) * normal));
}

__device__ const bool isZero(const glm::vec3 * in) {
    return ((fabs(in->x) < epsil) && (fabs(in->y) < epsil) && (fabs(in->z) < epsil)); 
}

__host__ uint8_t generateRandomNumH() {
    return static_cast<uint8_t>(rand() % 255);
}

__host__ float generateRandomFloatH() {
    return (static_cast<float>(std::rand() - RAND_MAX/2)/RAND_MAX);
}
__device__ float generateRandomFloatD(curandState * const state) {
    return curand_uniform(state) - .500000000001f;
}   

__device__ float generateNormalFloatD(curandState * const state) {
    return curand_normal(state);
}   

/*
* The point of this is to get a y value from a Z score, with
* the higher the z score, the lower the return value
* if fabs( rand value) < epsil, return 1/epsil to avoid any rounding errors
* returning 100000f isnt a great solution, but it will do
*/
__device__ float generateInvNormalFloatD(curandState * const state) {
    float rand = generateNormalFloatD(state);
    return fabs(rand)<epsil?1000000.0f:1.0f/rand;
}  



__host__ glm::vec3 generateRandomVecH() {
    return glm::normalize(glm::vec3(
        generateRandomFloatH(), generateRandomFloatH(), generateRandomFloatH()));
}
__device__ glm::vec3 generateRandomVecD(curandState * const state) {
    return glm::normalize(glm::vec3(
        generateRandomFloatD(state), generateRandomFloatD(state), generateRandomFloatD(state)));
}

__device__ glm::vec3 generateNormalVecD(curandState * state) {
    return glm::normalize(glm::vec3(
        generateNormalFloatD(state), generateNormalFloatD(state), generateNormalFloatD(state)));
}

__device__ glm::vec3 generateInvNormalVecD(curandState * state) {
    return glm::normalize(glm::vec3(
        generateInvNormalFloatD(state), generateInvNormalFloatD(state), generateInvNormalFloatD(state)));
}