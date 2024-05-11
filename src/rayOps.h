#pragma once

#include <utility>
#include <math.h>
#include <limits.h>

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "./Color.h"
#include "./Object.h"
#include "./Primitives.h"

const int WIDTH = 400;
const int HEIGHT = 400;

const int FOV_Y = 60;
const int FOV_X = 90;

const int SPP = 4;
const int BOUNCES = 16;

const float ASPECT_RATIO = static_cast<float>(WIDTH)/HEIGHT;
const float epsil = .000001;
const float infinity = std::numeric_limits<float>::infinity();

using evalInfo = std::pair<bool, CollisionInfo>;



Color escape(const glm::vec3 & dir) {
    return Color(dir.x * dir.x, dir.y * dir.y, dir.z * dir.z);
}

bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & Edge1, const glm::vec3 & Edge2, CollisionInfo * out)
{
    glm::vec3 P = glm::cross(ray.Dir, Edge2);
    float determinant = glm::dot(P, Edge1);

    //NON BACKFACE CULLING
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

bool intersectsMesh(const Mesh & mesh, const Ray & ray, CollisionInfo * closestFace) 
{
    bool changed = false;
    for(size_t i = 0; i < mesh.Faces.size(); ++i) {
        const glm::vec3 PointA = mesh.Indicies[mesh.Faces[i].x].Pos;
        const glm::vec3 Edge1 = mesh.EdgeMap[(i * 2)]; //edge one
        const glm::vec3 Edge2 = mesh.EdgeMap[(i * 2) + 1]; //edge two
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
CollisionInfo checkCollisions(Ray & ray, const std::vector<Mesh> & meshs) //need scene. 
{
    float distance = 0;
    CollisionInfo closestObj;
    for (size_t i = 0; i < meshs.size();++i) {
        if (intersectsMesh(meshs[i], ray, &closestObj)) {
            closestObj.meshIdx = static_cast<int>(i);
        }
    }
    return closestObj;
}

Color eval(Ray & ray, const std::vector<Mesh> & objs, const int bounceCount) {
    if (bounceCount <= 0) {
        return Color(0, 0, 0);
    }

    CollisionInfo collide = checkCollisions(ray, objs);
    if (collide.meshIdx == -1) {
        float a = (.5 * (ray.Dir.y + 1.0));

        return Color(1, 1, 1) * (1-a)  + (Color(.5, .7, 1.0) * a);
    }

    return Color(255, 255, 255);
    //return objs[collide.objIdx].getMeshColor(collide);
    //const Object & curObj = objs[collide.objIdx];

    const Mesh & curMesh = objs[collide.meshIdx]; // curObj.getObjInfo()[collide.meshIdx];
    const glm::vec3 & normal = curMesh.getFaceNormal(collide.faceIdx);
    float lambertian = curMesh.getMaterial().getLambertian(ray, normal);
    //Color scatter = curMesh.getColor

    //random direction
    glm::vec3 newOrigin = ray.Origin + collide.distanceMin * ray.Dir;
    //Ray newRay = curObj.generateRandomVecOnFace(collide.meshIdx, collide.faceIdx, newOrigin);
    //return  (eval(newRay, objs, bounceCount -1) * .5f);
}

/*
* TraceRay is responsible for creating the ray and giving it a direction based on u,v.
* Takes in the objects to determine collisions.
*/
Color traceRay(float u, float v, const std::vector<Mesh> & objs)
{
    Ray ray;
    ray.Origin = glm::vec3(0, 0, 0); //model-world-camera conversion.
    ray.Dir = glm::vec3(u, v, 1.0f);
    normalizeRayDir(ray);

    return eval(ray, objs, BOUNCES);
}


/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x, const std::vector<Mesh> & objs)
{   
    Color Final;
    const float delta_u = ASPECT_RATIO * 1.0f/(WIDTH);
    const float delta_v = 1.0f/(HEIGHT);
    for(size_t i = 0; i < static_cast<size_t>(SPP); ++i)
    {
        float u =  (ASPECT_RATIO) * (static_cast<float>(x) - (WIDTH/2.0))/WIDTH; //convert from x,y to -.5, .5
        float v = (static_cast<float>(y) - (HEIGHT/2.0))/HEIGHT; 
        
        //ANTI ALIASING!
        //float varU = generateRandomFloat() * delta_u;
        //float varV = generateRandomFloat() * delta_v;
        //u += varU;
        //u += varV;

        Final += traceRay(u, v, objs);
    }
    return Final/static_cast<float>(SPP);
}