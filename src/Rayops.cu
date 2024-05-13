#include "rayOps.cuh"

Color escape(const glm::vec3 & dir) {
    return Color(dir.x * dir.x, dir.y * dir.y, dir.z * dir.z);
}

__device__ bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & Edge1, const glm::vec3 & Edge2, CollisionInfo * out)
{
    glm::vec3 P = glm::cross(ray.Dir, Edge2);
    float determinant = glm::dot(P, Edge1);

    //BACKFACE CULLING
    if (determinant < epsil) {
        return false;
    }

    float inv_det = 1.0/determinant;
    glm::vec3 T = ray.Origin - PointA;
    float u_bar = glm::dot(P, T) * inv_det;
    if (u_bar < 0 || u_bar > 1) {
        return false;
    }

    glm::vec3 Q = glm::cross(T, Edge1);
    float v_bar = glm::dot(Q, ray.Dir) * inv_det;
    if (v_bar < 0 ||  (v_bar + u_bar) > 1) {
        return false;
    }
    float t_bar = glm::dot(Q, Edge2) * inv_det;
    if(t_bar < 0 || !(t_bar < out->distanceMin)) {
        return false;
    }

    out->distanceMin = t_bar;
    out->CollisionPoint.x = u_bar;
    out->CollisionPoint.y = v_bar;
    return true; 
}

__device__ bool intersectsMesh(const MeshGpu * mesh, const Ray & ray, CollisionInfo * closestFace) 
{
    bool changed = false;
    for(size_t i = 0; i < mesh->faceSize; ++i) {
        const glm::vec3 & PointA = mesh->vertexBuffer[mesh->faceBuff[i].x].Pos;
        const glm::vec3 & Edge1 = mesh->edgeBuff[(i * 2)]; //edge one
        const glm::vec3 & Edge2 = mesh->edgeBuff[(i * 2) + 1]; //edge two

        if(intersectsTri(ray, PointA, Edge1, Edge2, closestFace)) {
            changed = true;
            closestFace->faceIdx = static_cast<int>(i);
        }
    }
    return changed;
}

/*
* Responsible for checking if a ray collides with an object for all objects in the scene. 
*/
__device__ CollisionInfo checkCollisions(const Ray & ray, const GpuInfo * info) {
    CollisionInfo closestObj;
    for (size_t i = 0; i < info->meshLen;++i) {
        if (intersectsMesh(&(info->meshDev[i]), ray, &closestObj)) {
            closestObj.meshIdx = static_cast<int>(i);
        }
    }
    return closestObj;
}


/*
* Doesnt work due to ray not changing
* also need to figure out the factor portion. 
*/
__device__ Color evalIter(Ray & ray, const GpuInfo * info, curandState * const randState, const int bounceCount) {
    Color final = Color(0.0f, 0.0f, 0.0f);
    CollisionInfo collide;
    const MeshGpu * curMesh;
    const glm::vec3 * normal;
    float factor = 1.0f;
    for (size_t i = 0; i < static_cast<size_t>(bounceCount); ++i) {
        collide = checkCollisions(ray, info);
        if (collide.meshIdx == -1) {
            float a = (.5 * (ray.Dir.y + 1.0));
            final += (Color(1, 1, 1) * (1-a)  + (Color(.5, .7, 1.0) * a)) * factor;
            return final;
        }

        curMesh = &(info->meshDev[collide.meshIdx]);
        //normal = &(curMesh->getFaceNormal(collide.faceIdx));

        // random direction
        glm::vec3 newOrigin = ray.Origin + collide.distanceMin * ray.Dir;
        ray = curMesh->generateReflectiveVecOnFace(collide.faceIdx, ray.Dir, newOrigin);
        collide.meshIdx = -1;
        factor *= .5f;
    }
    
    return Color(0.0f, 0.0f, 0.0f); //bounce count exceeded
}

/*
 * Not usable; results in cuda kernel error bc of requesting too many resources
*/
__device__ Color eval(Ray & ray, const GpuInfo * info, curandState * const randState, const int bounceCount) {
    if (bounceCount <= 0) {
        return Color(0, 0, 0);
    }

    CollisionInfo collide = checkCollisions(ray, info);
    if (collide.meshIdx == -1) {
        float a = (.5 * (ray.Dir.y + 1.0));

        return Color(1, 1, 1) * (1-a)  + (Color(.5, .7, 1.0) * a);
    }

    const MeshGpu & curMesh = info->meshDev[collide.meshIdx];
    const glm::vec3 & normal = curMesh.getFaceNormal(collide.faceIdx);

    //random direction
    glm::vec3 newOrigin = ray.Origin + collide.distanceMin * ray.Dir;
    Ray newRay = curMesh.generateLambertianVecOnFace(collide.faceIdx, randState, newOrigin);
    return  (eval(newRay, info, randState, bounceCount -1) * .5f);
}

/*
* TraceRay is responsible for creating the ray and giving it a direction based on u,v.
* Takes in the objects to determine collisions.
*/
__device__ Color traceRay(float u, float v, curandState * const randState, GpuInfo * info) {
    Ray ray;
    ray.Origin = glm::vec3(0, 0, 0); 
    ray.Dir = glm::vec3(u, v, 1.0f);
    normalizeRayDir(ray);
    return evalIter(ray, info, randState, BOUNCES);
}


/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
__global__ void spawnRay(GpuInfo info, int seed, uint8_t * colorArr) {   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int one_d_idx = CHANNEL * ((idy  * WIDTH) + (idx));

    if(idx >= WIDTH || idy >= HEIGHT) {
        return;
    } 

    const float delta_u = ASPECT_RATIO * 1.0f/(WIDTH); 
    const float delta_v = 1.0f/(HEIGHT);
    curandState randState;
    curand_init(seed, one_d_idx ,0, &randState);

    Color Final;

    for (size_t i = 0; i < static_cast<size_t>(SPP); ++i) {
        float u =  (ASPECT_RATIO) * (static_cast<float>(idx) - (WIDTH/2.0))/WIDTH; 
        float v = (static_cast<float>(idy) - (HEIGHT/2.0))/HEIGHT; 
        
        // ANTI ALIASING!
        u += generateRandomFloatD(&randState) * delta_u;
        u += generateRandomFloatD(&randState) * delta_v;

        Final += traceRay(u, v, &randState, &info);
    }

    /*
    * Seperate so that we can gamma correct easier
    */
    Final /= static_cast<float>(SPP);
    gammaCorrect(&Final);
    Final *= 255.0f;
    Final = clampColor(Final);


    colorArr[one_d_idx] = Final.r;
    colorArr[one_d_idx + 1] = Final.g;
    colorArr[one_d_idx + 2] = Final.b;
}


/*
* 
* Achieves the exact same as spawnRay, but does so progressively so that we can write to the image
*/
__global__ void spawnRayProgressive(GpuInfo info, int seed, float * colorArr) {   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int one_d_idx = CHANNEL * ((idy  * WIDTH) + (idx));

    if(idx >= WIDTH || idy >= HEIGHT) {
        return;
    } 

    const float delta_u = ASPECT_RATIO * 1.0f/(WIDTH); 
    const float delta_v = 1.0f/(HEIGHT);
    curandState randState;
    curand_init(seed, one_d_idx, 0, &randState);

    float u =  (ASPECT_RATIO) * (static_cast<float>(idx) - (WIDTH/2.0))/WIDTH; 
    float v = (static_cast<float>(idy) - (HEIGHT/2.0))/HEIGHT; 

    u += generateRandomFloatD(&randState) * delta_u;
    u += generateRandomFloatD(&randState) * delta_v;

    Color Final = traceRay(u, v, &randState, &info);

    Final /= static_cast<float>(SPP);

    colorArr[one_d_idx] += Final.r;
    colorArr[one_d_idx + 1] += Final.g;
    colorArr[one_d_idx + 2] += Final.b;
}

/*
* Convert the color array of floats to uint8_t.
* Applies sqrt to gamma correct.
*/
__device__ void converColorProgressive(const float * num, uint8_t *out) {
    out[0] = num[0]>1?255:static_cast<uint8_t>(255 * sqrt(num[0]));
    out[1] = num[1]>1?255:static_cast<uint8_t>(255 * sqrt(num[1]));
    out[2] = num[2]>1?255:static_cast<uint8_t>(255 * sqrt(num[2]));
}

/*
* Convert the float array to a uint8_t array
*/
__global__ void convertArr(float * colorArr, uint8_t * out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int one_d_idx = CHANNEL * ((idy  * WIDTH) + (idx));
    
    if(idx >= WIDTH || idy >= HEIGHT) {
        return;
    } 

    converColorProgressive(&colorArr[one_d_idx], &out[one_d_idx]);
}

/*
* Wipe array to prepare for writing
*/
__global__ void wipeArr(float * colorArr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int one_d_idx = CHANNEL * ((idy  * WIDTH) + (idx));
    if(idx >= WIDTH || idy >= HEIGHT) {
        return;
    } 
    colorArr[one_d_idx] = 0;
    colorArr[one_d_idx + 1] = 0;
    colorArr[one_d_idx + 2] = 0;
}

/*
* Correct gamma
*/
__device__ void gammaCorrect(Color * colorPtr) {
    colorPtr->r = sqrtf(colorPtr->r);
    colorPtr->g = sqrtf(colorPtr->g);
    colorPtr->b = sqrtf(colorPtr->b);
}

