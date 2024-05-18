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

__device__ glm::vec3 MatGpu::samplePoint(const CollisionInfo * hitLoc, const curandState_t * state) const {
    glm::vec2 idx = getIdx(hitLoc);
    const float * specColor = getTextureColor(TextureArr[2], &idx);
    const float * MetalColor = getTextureColor(TextureArr[3], &idx);
    const float * roughChance = getTextureColor(TextureArr[4], &idx);

    const float specular = (1.0f - (*MetalColor)) * (1.0f - (*MetalColor));
    const float dielectric = (1.0f - (*MetalColor)) * (1.0f - (*specColor));

    const float specularWeight = *MetalColor + dielectric;



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

__device__ size_t MatGpu::getTextureIdx(const TextInfo * inf, const glm::vec2 * idx) const {
    if(inf->basic) {
        return 0;
    }
    size_t xInt = static_cast<size_t>(inf->width * idx->x);
    size_t yInt = static_cast<size_t>(inf->height * idx->y);
    return (CHANNEL * (yInt * inf->width + xInt));
}

__device__ const float * MatGpu::getTextureColor(const TextInfo * inf, const glm::vec2 * idx) const {
    if(inf->basic) {
        return inf->basicColor;
    }
    size_t xInt = static_cast<size_t>(inf->width * idx->x);
    size_t yInt = static_cast<size_t>(inf->height * idx->y);
    return inf->arr + (CHANNEL * (yInt * inf->width + xInt));
}

/*
* Bad : getIdx was designed around an image with 3 channels
* how is roughness stored?
* is it 1 channel or 3?
* deference assumes it is 3 channel
* also : what is basecolor in this context?
*/
__device__ const float * MatGpu::colorAt(const CollisionInfo * hitLoc, const shaderInfo * info) const {
    glm::vec2 idx = getIdx(hitLoc);
    return getTextureColor(TextureArr[0], &idx);
}   

/*
Bad : getIdx was designed around an image with 3 channels
* how is roughness stored? is it 1 channel or 3? deference assumes it is 3 channel
*/
__device__ float MatGpu::baseDiffuse(const glm::vec2 * idx, const shaderInfo * info) const {
    const float FD90 = .5f + (2 * (*getTextureColor(TextureArr[4], idx)) * (info->hdotw_out * info->hdotw_out));
    const float FDWOUT = 1 + ((FD90 - 1) * (1 - info->ndotw_out_pow5));
    const float FDWIN = 1 + ((FD90 - 1) * (1 - info->ndotw_in_pow5));
    const float baseDiff = (1/pi) * FDWIN * FDWOUT * info->ndotw_out;
    return baseDiff;
}

__device__ float MatGpu::baseSubsurface(const glm::vec2 * idx, const shaderInfo * info) const {
    const float FSS90 = *getTextureColor(TextureArr[4], idx) * (info->hdotw_out * info->hdotw_out);
    const float FSSWOUT = 1 + ((FSS90 - 1) * (1 - info->ndotw_out_pow5));
    const float FSSWIN = 1 + ((FSS90 - 1) * (1 - info->ndotw_in_pow5));
    const float VOLUMEABSORB = (1.0f/(info->ndotw_in + info->ndotw_out)) - .5f;
    const float baseSS = (1.25f/pi) * (FSSWIN * FSSWOUT * VOLUMEABSORB + .5f) * info->ndotw_out;
    return baseSS;
}

__device__ glm::vec3 MeshGpu::generateRandomVecOnFace(const size_t faceIdx, curandState * state) const {
    const glm::vec3 newDir = getFaceNormal(faceIdx) + generateRandomVecD(state);
    if (isZero(&newDir)) {
        return getFaceNormal(faceIdx);
    }
    return glm::normalize(newDir);
}

__device__ glm::vec3 MeshGpu::generateReflectiveVecOnFace(const size_t faceIdx, const glm::vec3 & dir) const {
    const glm::vec3 & normal = getFaceNormal(faceIdx);
    const glm::vec3 newDir = dir - (2 * glm::dot(normal, dir) * normal);
    if (isZero(&newDir)) {
        return getFaceNormal(faceIdx);
    }
    return glm::normalize(newDir);
}

__device__ glm::vec3 MeshGpu::generateRoughVecOnFace(const size_t faceIdx, const glm::vec3 & dir) const {
    const glm::vec3 & normal = getFaceNormal(faceIdx);
    const glm::vec3 newDir = dir - (2 * glm::dot(normal, dir) * normal);
    if (isZero(&newDir)) {
        return getFaceNormal(faceIdx);
    }
    return glm::normalize(newDir);
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