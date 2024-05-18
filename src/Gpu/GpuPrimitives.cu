#include "GpuPrimitives.cuh"


__device__ void shaderInfo::setRequired(const Ray * rayIn, const Ray * rayOut, const glm::vec3 * normal){
    h = glm::normalize(rayIn->Dir + rayOut->Dir);
   
    ndotw_in = fabs(glm::dot(*normal, rayIn->Dir));
    ndotw_out = fabs(glm::dot(*normal, rayOut->Dir));
    hdotw_in = fabs(glm::dot(h, rayIn->Dir));
    hdotw_out = fabs(glm::dot(h, rayOut->Dir));

    ndotw_in_pow5 = powf(ndotw_in,5);
    ndotw_out_pow5 = powf(ndotw_out, 5);

}

 __device__ glm::vec2 MatGpu::getIdx(const CollisionInfo * hitLoc) const {
    const float u = hitLoc->CollisionPoint.x;
    const float v = hitLoc->CollisionPoint.y;
    const float w = 1 - (hitLoc->CollisionPoint.x + hitLoc->CollisionPoint.y);
    const glm::vec2 * PointA = hitLoc->TQA;
    const glm::vec2 * PointB = hitLoc->TQB;
    const glm::vec2 * PointC = hitLoc->TQC;

    float idx = (w * PointA->x + u * PointB->x + v * PointC->x); 
    float idy = (w * PointA->y + u * PointB->y + v * PointC->y);
    
   return glm::vec2(idx, idy);
}

 __device__ size_t MatGpu::getTextureIdx(TextInfo * inf, glm::vec2 * idx) const {
    size_t xInt = static_cast<size_t>((inf->width * idx->x) + .5f);
    size_t yInt = static_cast<size_t>((inf->height * idx->y) + .5f);
    return (CHANNEL * (yInt * inf->width + xInt));
 }

 __device__ float * MatGpu::getTextureColor(TextInfo * inf, glm::vec2 * idx) const {
    size_t xInt = static_cast<size_t>((inf->width * idx->x) + .5f);
    size_t yInt = static_cast<size_t>((inf->height * idx->y) + .5f);
    return inf->arr + (CHANNEL * (yInt * inf->width + xInt));
 }

/*
* Bad : getIdx was designed around an image with 3 channels
* how is roughness stored?
* is it 1 channel or 3?
* deference assumes it is 3 channel
* also : what is basecolor in this context?
*/
__device__ float * MatGpu::colorAt(const CollisionInfo * hitLoc, const shaderInfo * info) const {
    glm::vec2 idx = (getIdx(hitLoc));
    float diffFactor = baseDiffuse(&idx, info);
    float subsurface = baseSubsurface(&idx, info);
}   

/*
Bad : getIdx was designed around an image with 3 channels
* how is roughness stored? is it 1 channel or 3? deference assumes it is 3 channel
*/
__device__ float MatGpu::baseDiffuse(glm::vec2 * idx, const shaderInfo * info) const {
    const float FD90 = .5f + (2 * (*getTextureColor(TextureArr[4], idx)) * (info->hdotw_out * info->hdotw_out));
    const float FDWOUT = 1 + ((FD90 - 1) * (1 - info->ndotw_out_pow5));
    const float FDWIN = 1 + ((FD90 - 1) * (1 - info->ndotw_in_pow5));
    const float baseDiff = (1/pi) * FDWIN * FDWOUT * info->ndotw_out;
}

__device__ float MatGpu::baseSubsurface(glm::vec2 * idx, const shaderInfo * info) const {
    const float FSS90 = *getTextureColor(TextureArr[4], idx) * (info->hdotw_out * info->hdotw_out);
    const float FSSWOUT = 1 + ((FSS90 - 1) * (1 - info->ndotw_out_pow5));
    const float FSSWIN = 1 + ((FSS90 - 1) * (1 - info->ndotw_in_pow5));
    const float VOLUMEABSORB = (1.0f/(info->ndotw_in + info->ndotw_out)) - .5f;
    const float baseDiff = (1.25f/pi) * (FSSWIN * FSSWOUT * VOLUMEABSORB + .5f) * info->ndotw_out;
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