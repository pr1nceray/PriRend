#include "GpuPrimitives.cuh"

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
