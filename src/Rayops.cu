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
    float distance = 0;
    CollisionInfo closestObj;
    for (size_t i = 0; i < info->meshLen;++i) {
        if (intersectsMesh(&(info->meshDev[i]), ray, &closestObj)) {
            closestObj.meshIdx = static_cast<int>(i);
        }
    }
    return closestObj;
}

__device__ Color eval(Ray & ray, const GpuInfo * info, const int bounceCount) {
    if (bounceCount <= 0) {
        return Color(0, 0, 0);
    }

    CollisionInfo collide = checkCollisions(ray, info);
    if (collide.meshIdx == -1) {
        float a = (.5 * (ray.Dir.y + 1.0));

        return Color(0,0,0);//Color(1, 1, 1) * (1-a)  + (Color(.5, .7, 1.0) * a);
    }
    else {
        return Color(1,1,1);
    }

    const MeshGpu & curMesh = info->meshDev[collide.meshIdx];
    const glm::vec3 & normal = curMesh.getFaceNormal(collide.faceIdx);

    //random direction
    glm::vec3 newOrigin = ray.Origin + collide.distanceMin * ray.Dir;
    Ray newRay = curMesh.generateLambertianVecOnFace(collide.faceIdx, newOrigin);
    return  (eval(newRay, info, bounceCount -1) * .5f);
}

/*
* TraceRay is responsible for creating the ray and giving it a direction based on u,v.
* Takes in the objects to determine collisions.
*/
__device__ Color traceRay(float u, float v, const GpuInfo * info) {
    Ray ray;
    ray.Origin = glm::vec3(0, 0, 0); 
    ray.Dir = glm::vec3(u, v, 1.0f);
    normalizeRayDir(ray);

    return eval(ray, info, BOUNCES);
}

/*
* Clamp the color to be a valid color.
*/
__device__ Color clampColor(Color final) {
    uint8_t finalR = final.r > 255? 255 : static_cast<uint8_t>(final.r);
    uint8_t finalG = final.g > 255? 255 : static_cast<uint8_t>(final.g);
    uint8_t finalB = final.b > 255? 255 : static_cast<uint8_t>(final.b);
    return Color(finalR, finalG, finalB);
}



/*

TODO : get the seed as input from camera (should use a rand() call there)
TODO2 : pass rand state in to traceRay/eval, 
TODO3 : use CUrandState in generateRandom()
TODO4 : debug. 

*/


/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
__global__ void spawnRay(GpuInfo info, uint8_t * colorArr) {   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int one_d_idx = (idy * WIDTH * CHANNEL) + idx;

    if(idx > WIDTH || idy > WIDTH) {
        return;
    } 

    const float delta_u = ASPECT_RATIO * 1.0f/(WIDTH); 
    const float delta_v = 1.0f/(HEIGHT);
    curandState randState;

    curand_init(3, one_d_idx ,0, &randState);
    Color Final;

    for (size_t i = 0; i < static_cast<size_t>(SPP); ++i) {
        float u =  (ASPECT_RATIO) * (static_cast<float>(idx) - (WIDTH/2.0))/WIDTH; 
        float v = (static_cast<float>(idy) - (HEIGHT/2.0))/HEIGHT; 
        
        //ANTI ALIASING!
        float varU = generateRandomFloatD() * delta_u;
        float varV = generateRandomFloatD() * delta_v;
        u += varU;
        u += varV;

        Final += traceRay(u, v, &info);
    }

    Final /= static_cast<float>(SPP);
    Final = clampColor(Final);

    colorArr[one_d_idx] = Final.r;
    colorArr[one_d_idx + 1] = Final.g;
    colorArr[one_d_idx + 2] = Final.b;

    
    //return Final/static_cast<float>(SPP);
}