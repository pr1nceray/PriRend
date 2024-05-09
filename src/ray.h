#pragma once
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include "Color.h"
#include "object.h"
#include "Triangle_Ops.h"

const int WIDTH = 300;
const int HEIGHT = 200;

const int FOV_Y = 60;
const int FOV_X = 90;

const float ASPECT_RATIO = WIDTH/HEIGHT;

const int SPP = 16;

const float epsilon = .0000005;

struct Ray
{
    glm::vec3 Origin;
    glm::vec3 Dir;
};

void normalizeRayDir(Ray & ray)
{
    ray.Dir = glm::normalize(ray.Dir);
}


bool intersectsTri(const Ray & ray, const glm::vec3 & PointA,
                 const glm::vec3 & PointB, const glm::vec3 & PointC)
{
    //SHOULD BE DONE ON OBJECT CREATION
    //NOT REQUIRED TO BE DONE DURING RENDERTIME
    Ray normal = Ray(); //obtainNormal(PointA, PointB, PointC);

    float n_dot_ray_origin = glm::dot(normal.Dir, ray.Origin);
    float n_dot_ray_Dir= glm::dot(normal.Dir, ray.Dir);

    if(abs(n_dot_ray_origin) < epsilon)
    {
        //int the case of perpendicularity, return true for simplicity
        return true;
    }
    
    //need to solve for constant D
    return false;

}
bool intersectsMesh(const Mesh & mesh, const Ray & ray) 
{
    for(size_t i = 0; i < mesh.Faces.size(); ++i)
    {
        
        const glm::vec3 PointA = mesh.Indicies[mesh.Faces[i].x].Pos;
        const glm::vec3 PointB = mesh.Indicies[mesh.Faces[i].y].Pos;
        const glm::vec3 PointC = mesh.Indicies[mesh.Faces[i].z].Pos;

        if(intersectsTri(ray, PointA, PointB, PointC))
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
    for(size_t i = 0; i < objs.size();++i)
    {
        if(intersectsOBJ(objs[i], ray))
        {
            //for now, a constant color if we intersect with 
            //an object
            return Color(125, 125, 125); 
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
    ray.Origin = glm::vec4(0, 0, 0, 1); //model-world-camera conversion.
    ray.Dir = glm::vec4(u, v, 1.0f, 0.0f);
    normalizeRayDir(ray);

    return checkCollisions(ray, objs);
}
\
/*
* spawnRay is responsible for creating parameters and calling traceRay
* Averages the findings of the samples (controlled by SPP), and returns a color.
*/
Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x, const std::vector<object> & objs)
{   
    Color Final;
    for(size_t i = 0; i < SPP; ++i)
    {
        float u = ASPECT_RATIO * (static_cast<float>(x) - (WIDTH/2.0))/WIDTH; //convert from x,y to -.5, .5
        float v = (static_cast<float>(y) - (HEIGHT/2.0))/HEIGHT; 
        if(i == 0)
        {
            std::cout << "ray dir : " << u << " , " << v << "\n";
        }
        
        Final += traceRay(u, v, objs);
    }
    return Color{255, 255, 255 };
}