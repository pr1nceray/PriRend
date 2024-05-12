#include "Primitives.h"

 __host__ __device__ void normalizeRayDir(Ray & ray) {
    ray.Dir = glm::normalize(ray.Dir);
}

/*
* Flip a ray along the normal for PURE reflective materials. 
*/
glm::vec3 flipRayNormal(const Ray & ray, const glm::vec3 & normal) {
    return ray.Dir - (2.0f * (glm::dot(ray.Dir, normal) * normal));
}