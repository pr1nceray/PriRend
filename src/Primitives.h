#pragma once
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.h"

struct Vertex
{
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;

};

struct Ray
{
    glm::vec3 Origin;
    glm::vec3 Dir;
};
