#include "GpuPrimitives.cuh"

/*
* Chunky function ngl
*/
__device__ float * MatGpu::diffuseAtPoint(const CollisionInfo * hitLoc, const glm::vec2 * PointA,
    const glm::vec2 * PointB, const glm::vec2 * PointC) const {
    const float u = hitLoc->CollisionPoint.x;
    const float v = hitLoc->CollisionPoint.y;
    const float w = 1 - hitLoc->CollisionPoint.x - hitLoc->CollisionPoint.y;

    float idx = (w * PointA->x + u * PointB->x + v * PointC->x) * WIDTH;
    float idy = (w * PointA->y + u * PointB->y + v * PointC->y)  * HEIGHT;

    int one_d_idx = static_cast<int>(idy * diffuse->width * CHANNEL + idx);
    return (diffuse->arr + one_d_idx);
}

__device__ Ray MeshGpu::generateRandomVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const {
    glm::vec3 randVec = generateRandomVecD(state);
    glm::vec3 normal = getFaceNormal(faceIdx);
    randVec *= glm::dot(randVec, normal) < 0?-1:1;

    glm::vec3 newOrigin = origin + (randVec * .001f); //avoid shadow acne
    return Ray(newOrigin, randVec);
}

__device__ Ray MeshGpu::generateLambertianVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const {
    glm::vec3 newDir = getFaceNormal(faceIdx) + generateRandomVecD(state);
    glm::vec3 newOrigin = origin + (newDir * .01f); // avoid shadow acne
    return Ray(newOrigin, newDir);
}

__device__ Ray MeshGpu::generateReflectiveVecOnFace(const size_t faceIdx, const glm::vec3 & dir, const glm::vec3 & origin) const {
    const glm::vec3 & normal = getFaceNormal(faceIdx);
    const glm::vec3 newDir = dir - (2 * glm::dot(normal, dir) * normal);
    const glm::vec3 newOrigin = origin + (newDir * .01f); // avoid shadow acne
    return Ray(newOrigin, newDir);
}
