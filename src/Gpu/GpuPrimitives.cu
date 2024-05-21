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
    const float4 specColor = getTextureColor(TextureArr[2], &idx);
    const float4 MetalColor = getTextureColor(TextureArr[3], &idx);
    const float4 roughChance = getTextureColor(TextureArr[4], &idx);

    const float specular = (1.0f - (MetalColor.x)) * (1.0f - (MetalColor.x));
    const float dielectric = (1.0f - (MetalColor.x)) * (1.0f - (specColor.x));

    const float specularWeight = MetalColor.x + dielectric;



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


__device__ const float4 MatGpu::getTextureColor(const TextInfo * inf, const glm::vec2 * idx) const {
    if(inf->basic) {
        return inf->basicColor;
    }
    return tex2D<float4>(inf->text, idx->x, idx->y);
}

/*
* Bad : getIdx was designed around an image with 3 channels
* how is roughness stored?
* is it 1 channel or 3?
* deference assumes it is 3 channel
* also : what is basecolor in this context?
*/
__device__ const float4 MatGpu::colorAt(const CollisionInfo * hitLoc, const shaderInfo * info) const {
    glm::vec2 idx = getIdx(hitLoc);
    return getTextureColor(TextureArr[0], &idx);
}   

/*
Bad : getIdx was designed around an image with 3 channels
* how is roughness stored? is it 1 channel or 3? deference assumes it is 3 channel
*/
__device__ float MatGpu::baseDiffuse(const glm::vec2 * idx, const shaderInfo * info) const {
    const float FD90 = .5f + (2 * (getTextureColor(TextureArr[4], idx).x) * (info->hdotw_out * info->hdotw_out));
    const float FDWOUT = 1 + ((FD90 - 1) * (1 - info->ndotw_out_pow5));
    const float FDWIN = 1 + ((FD90 - 1) * (1 - info->ndotw_in_pow5));
    const float baseDiff = (1/pi) * FDWIN * FDWOUT * info->ndotw_out;
    return baseDiff;
}

__device__ float MatGpu::baseSubsurface(const glm::vec2 * idx, const shaderInfo * info) const {
    const float FSS90 = getTextureColor(TextureArr[4], idx).x * (info->hdotw_out * info->hdotw_out);
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
    printf("%d %d\n", 1024, 1024);
    printf("255\n");
    for(size_t i = 0; i < 1024; ++i) {
        for(size_t j = 0; j < 1024; ++j) {
            float idy = i/1024.0f;
            float idx = j/1024.0f;
            float4 color = tex2D<float4>(text->text, idx, idy);
            uint8_t r = color.x>1?255:static_cast<uint8_t>(255 * color.x);
            uint8_t g = color.y>1?255:static_cast<uint8_t>(255 * color.y);
            uint8_t b = color.z>1?255:static_cast<uint8_t>(255 * color.z);
            printf("%d %d %d ", r, g, b);

        }
        printf("\n");
    }
}
