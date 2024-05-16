#include "GpuPrimitives.cuh"

/*
* Chunky function ngl
*/
__device__ float *MatGpu::diffuseAtPoint(const CollisionInfo * hitLoc) const {
    const float u = hitLoc->CollisionPoint.x;
    const float v = hitLoc->CollisionPoint.y;
    const float w = 1 - (hitLoc->CollisionPoint.x + hitLoc->CollisionPoint.y);
    const glm::vec2 * PointA = hitLoc->TQA;
    const glm::vec2 * PointB = hitLoc->TQB;
    const glm::vec2 * PointC = hitLoc->TQC;

    size_t idx = static_cast<size_t>(((w * PointA->x + u * PointB->x + v * PointC->x) * diffuse->width) + .5f);
    size_t idy = static_cast<size_t>(((w * PointA->y + u * PointB->y + v * PointC->y) * diffuse->height) + .5f);
    
    int one_d_idx = (idy * diffuse->width + idx) * CHANNEL;
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


__device__ void printTextures(TextInfo * text) {
    printf("P3\n");
    printf("%d %d\n", text->width, text->height);
    printf("255\n");
    for(size_t i = 0; i < text->height; ++i) {
        for(size_t j = 0; j < text->width; ++j) {
            size_t idx = CHANNEL * ((i * text->width) + j);
            uint8_t r = text->arr[idx]>1?255:static_cast<uint8_t>(255 * text->arr[idx]);
            uint8_t g = text->arr[idx + 1]>1?255:static_cast<uint8_t>(255 * text->arr[idx + 1]);
            uint8_t b = text->arr[idx + 2]>1?255:static_cast<uint8_t>(255 * text->arr[idx + 2]);
            printf("%d %d %d ", r, g, b);

        }
        printf("\n");
    }
}