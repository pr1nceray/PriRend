#pragma once
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include "Color.h"

const int WIDTH = 300;
const int HEIGHT = 200;

const int FOV_Y = 60;
const int FOV_X = 90;

const float ASPECT_RATIO = WIDTH/HEIGHT;

const int SPP = 16;

struct Ray
{
    glm::vec4 Start;
    glm::vec4 Dir;
};


Color traceRay(float u, float v)
{
    Ray ray;
    ray.Start = glm::vec4(0, 0, 0, 1); //change to camera position
}

Color spawnRay(size_t x, size_t y, size_t fov_y, size_t fov_x)
{   
    Color Final;
    for(size_t i = 0; i < SPP; ++i)
    {
        float u = (static_cast<float>(x) - (WIDTH/2.0))/WIDTH; //convert from x,y to -.5, .5
        float v = (static_cast<float>(y) - (WIDTH/2.0))/HEIGHT; 
        Final += traceRay(u,v);
    }
    return Color{255, 255, 255 };
}