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
        //if (intersectsMesh(&(info->meshDev[i]), ray, &closestObj)) {
        //    closestObj.meshIdx = static_cast<int>(i);
        //}
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
    float factor = 1.0f;
    const MeshGpu * curMesh;
    const glm::vec3 * normal;
    for (size_t i = 0; i < static_cast<size_t>(bounceCount); ++i) {
        collide = checkCollisions(ray, info);
        if (collide.meshIdx == -1) {
            float a = (.5 * (ray.Dir.y + 1.0));
            final += (Color(1, 1, 1) * (1-a)  + (Color(.5, .7, 1.0) * a)) * factor;
            return Color(1.0f, 1.0f, 1.0f);
        }

        curMesh = &(info->meshDev[collide.meshIdx]);
        normal = &(curMesh->getFaceNormal(collide.faceIdx));

        // random direction
        glm::vec3 newOrigin = ray.Origin + collide.distanceMin * ray.Dir;
        ray = curMesh->generateLambertianVecOnFace(collide.faceIdx, randState, newOrigin);
        factor *= .5f; //factor gets reduced
        collide.meshIdx = -1;
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

    printf("One d idx : %d\n",one_d_idx );

    const float delta_u = ASPECT_RATIO * 1.0f/(WIDTH); 
    const float delta_v = 1.0f/(HEIGHT);
    curandState randState;
    curand_init(seed, one_d_idx ,0, &randState);

    Color Final;

    for (size_t i = 0; i < static_cast<size_t>(SPP); ++i) {
        float u =  (ASPECT_RATIO) * (static_cast<float>(idx) - (WIDTH/2.0))/WIDTH; 
        float v = (static_cast<float>(idy) - (HEIGHT/2.0))/HEIGHT; 
        
        // ANTI ALIASING!
        float varU = generateRandomFloatD(&randState) * delta_u;
        float varV = generateRandomFloatD(&randState) * delta_v;
        u += varU;
        u += varV;

        Final += traceRay(u, v, &randState, &info);
    }

    Final /= static_cast<float>(SPP);
    Final *= 255.0f;
    Final = clampColor(Final);

    colorArr[one_d_idx] = Final.r;
    colorArr[one_d_idx + 1] = Final.g;
    colorArr[one_d_idx + 2] = Final.b;
}
