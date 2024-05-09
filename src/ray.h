#pragma once
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include "Color.h"
#include "object.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;

const int FOV_Y = 60;
const int FOV_X = 90;

const float ASPECT_RATIO = WIDTH/HEIGHT;

const int SPP = 16;

struct Ray
{
    glm::vec4 Start;
    glm::vec4 Dir;
};

void normalizeRayDir(Ray & ray)
{
    float magnitude = sqrt((ray.Dir.x * ray.Dir.x) +  (ray.Dir.y * ray.Dir.y) + (ray.Dir.z * ray.Dir.z));
    ray.Dir = glm::vec4(ray.Dir.x/magnitude, ray.Dir.y/magnitude, ray.Dir.z/magnitude, 1.0f);
}


bool intersectsOBJ(const object & obj, const Ray & ray)
{
    return true;
}

Color checkCollisions(Ray & ray, const std::vector<object> & objs) //need scene. 
{
    for(size_t i = 0; i < objs.size();++i)
    {
        if(intersectsOBJ(objs[i], ray))
        {
            return Color(125, 125, 125);
        }
    }
    return Color(125, 125, 125);
}

Color traceRay(float u, float v, const std::vector<object> & objs)
{
    Ray ray;
    ray.Start = glm::vec4(0, 0, 0, 1); //model-world-camera conversion.
    ray.Dir = glm::vec4(u, v, 1.0f, 0.0f);
    normalizeRayDir(ray);

    return checkCollisions(ray, objs);
}

Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x, const std::vector<object> & objs)
{   
    Color Final;
    for(size_t i = 0; i < SPP; ++i)
    {
        float u = (static_cast<float>(x) - (WIDTH/2.0))/WIDTH; //convert from x,y to -.5, .5
        float v = (static_cast<float>(y) - (HEIGHT/2.0))/HEIGHT; 
        if(i == 0)
        {
            std::cout << "ray dir : " << u << " , " << v << "\n";
        }
        
        Final += traceRay(u, v, objs);
    }
    return Color{255, 255, 255 };
}