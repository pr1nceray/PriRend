#pragma once
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <math.h>


#include "Color.h"
#include "object.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;

const int FOV_Y = 60;
const int FOV_X = 90;

const float ASPECT_RATIO = static_cast<float>(WIDTH)/HEIGHT;

const int SPP = 1;

const float epsil = .000001;

struct Ray
{
    glm::vec3 Origin;
    glm::vec3 Dir;
};

void normalizeRayDir(Ray & ray)
{
    ray.Dir = glm::normalize(ray.Dir);
}

struct CollisionInfo
{
    bool is_hit;
    float distance;
    Color hit_color;

};

bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & Edge1, const glm::vec3 & Edge2)
{
    glm::vec3 P = glm::cross(ray.Dir, Edge2);

    float determinant = glm::dot(P, Edge1);

    //NON BACKFACE CULLING
    float inv_det = 1.0/determinant;
    if(abs(determinant) < epsil) 
    {
        return false;
    }

    glm::vec3 T = ray.Origin - PointA;
    glm::vec3 Q = glm::cross(T, Edge1);

    float u_bar = glm::dot(P, T) * inv_det;

    if(u_bar < 0 || u_bar > 1) 
    {
        return false;
    }

    float v_bar = glm::dot(Q, ray.Dir) * inv_det;

    if(v_bar < 0 ||  (v_bar + u_bar) > 1) 
    {
        return false;
    }

    //NOTE : CALCULATE T!
    float w_bar = 1 - u_bar - v_bar;


    return true;
}

bool intersectsMesh(const Mesh & mesh, const Ray & ray) 
{
    for(size_t i = 0; i < mesh.Faces.size(); ++i)
    {
        const glm::vec3 PointA = mesh.Indicies[mesh.Faces[i].x].Pos;
        const glm::vec3 Edge1 = mesh.EdgeMap[(i * 2)]; //edge one
        const glm::vec3 Edge2 = mesh.EdgeMap[(i * 2) + 1]; //edge two

        if(intersectsTri(ray, PointA, Edge1, Edge2))
        {
            return true;
        }
    }
    return false;
}

/*
* Responsible for checking if a ray collides with a specific object.
*/

bool intersectsOBJ(const object & obj, const Ray & ray)
{


    for(size_t i = 0; i < obj.getObjInfo().size(); ++i)
    {
        if(intersectsMesh(obj.getObjInfo()[i], ray)) 
        {
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
Color checkCollisions(Ray & ray, const std::vector<object> & objs) //need scene. 
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
Color traceRay(float u, float v, const std::vector<object> & objs)
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
Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x, const std::vector<object> & objs)
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