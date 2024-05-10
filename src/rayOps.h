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

const int WIDTH = 300;
const int HEIGHT = 200;

const int FOV_Y = 60;
const int FOV_X = 90;

const float ASPECT_RATIO = static_cast<float>(WIDTH)/HEIGHT;

const int SPP = 1;

const float epsil = .000001;

using CollisionInfo = std::pair<int, glm::vec4>;


void normalizeRayDir(Ray & ray)
{
    ray.Dir = glm::normalize(ray.Dir);
}



bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & Edge1, const glm::vec3 & Edge2, glm::vec4 * out)
{
    glm::vec3 P = glm::cross(ray.Dir, Edge2);
    float determinant = glm::dot(P, Edge1);

    //NON BACKFACE CULLING
    if (abs(determinant) < epsil) {
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

    float w_bar = 1 - u_bar - v_bar;
    out->x = glm::dot(Q, Edge2) * inv_det; //calculate T
    out->y = w_bar;
    out->z = u_bar;
    out->w = v_bar;
    return true;
}

CollisionInfo intersectsMesh(const Mesh & mesh, const Ray & ray) 
{
    int idx = -1;
    glm::vec4 collisionClosest(std::numeric_limits<float>::infinity(), 0, 0, 0);
    for(size_t i = 0; i < mesh.Faces.size(); ++i)
    {
        const glm::vec3 PointA = mesh.Indicies[mesh.Faces[i].x].Pos;
        const glm::vec3 Edge1 = mesh.EdgeMap[(i * 2)]; //edge one
        const glm::vec3 Edge2 = mesh.EdgeMap[(i * 2) + 1]; //edge two

        glm::vec4 collisionInfo;
        // Checks if intersect
        if(intersectsTri(ray, PointA, Edge1, Edge2, &collisionInfo)) {
            // If so, are we closer than before? if not, we can discard result.
            // Move up to take advantage of boolean short circuting
            if(collisionInfo.x < collisionClosest.x) {
                collisionClosest = collisionInfo;
                idx = i;
            }
        }
    }

    //return i, and collision info.
    return std::make_pair(idx, collisionClosest);
}

/*
* Responsible for checking if a ray collides with a specific object.
*/

bool intersectsOBJ(const Object & obj, const Ray & ray)
{
    int closestObj = -1;
    for(size_t i = 0; i < obj.getObjInfo().size(); ++i) {
        CollisionInfo nearestIntersect = intersectsMesh(obj.getObjInfo()[i], ray);
        if(nearestIntersect.first != -1) {
            //return true if we intersect with 
            //any of our meshes (may contain multiple)
            return true;
        }
    }
    return false;
}

/*
* Responsible for checking if a ray collides with an object for all objects in the scene. 
*/
Color checkCollisions(Ray & ray, const std::vector<Object> & objs) //need scene. 
{
    float distance = 0;
    for(size_t i = 0; i < objs.size();++i)
    {
        if(intersectsOBJ(objs[i], ray))
        {
            //for now, a constant color if we intersect with an object
            return Color(255, 255, 255); 
        }
    }
    return Color(0, 0, 0);
}

/*
* TraceRay is responsible for creating the ray and giving it a direction based on u,v.
* Takes in the objects to determine collisions.
*/
Color traceRay(float u, float v, const std::vector<Object> & objs)
{
    Ray ray;
    ray.Origin = glm::vec3(0, 0, 0); //model-world-camera conversion.
    ray.Dir = glm::vec3(u, v, 1.0f);
    normalizeRayDir(ray);

    return checkCollisions(ray, objs);
}


/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x, const std::vector<Object> & objs)
{   
    Color Final;
    for(size_t i = 0; i < SPP; ++i)
    {
        float u =  ASPECT_RATIO * (static_cast<float>(x) - (WIDTH/2.0))/WIDTH; //convert from x,y to -.5, .5
        float v = (static_cast<float>(y) - (HEIGHT/2.0))/HEIGHT; 
        
        Final += (traceRay(u, v, objs) * (1.0f/SPP));
    }
    return Final;
}