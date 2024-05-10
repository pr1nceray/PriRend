#pragma once
#include <time.h>
#include <stdlib.h>

#include <limits>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.h"

struct Vertex {
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;
};

struct CollisionInfo {
    glm::vec2 CollisionPoint;
    float distanceMin;
    int faceIdx;
    int meshIdx;
    int objIdx;

    CollisionInfo() :
    CollisionPoint(glm::vec2(0, 0)),
    distanceMin(std::numeric_limits<float>::infinity()),
    faceIdx(-1), meshIdx(-1), objIdx(-1) {
    }
};

struct Ray {
    glm::vec3 Origin;
    glm::vec3 Dir;

    Ray(const glm::vec3 & originIn, const glm::vec3 dirIn)  :
    Origin(originIn), Dir(dirIn) {
    }
};

/*
* Generate uniformly random float on the range -.5, .5
*/
inline float generateRandomFloat() {
    return (static_cast<float>(std::rand() - RAND_MAX/2)/RAND_MAX);
}

inline uint8_t generateRandomNum() {
    return static_cast<uint8_t>(rand() % 255);
}
/*
* Generates a random normalized vector
*/
inline glm::vec3 generateRandomVec() {
    return glm::normalize(glm::vec3(
        generateRandomFloat(), generateRandomFloat(), generateRandomFloat()));
}
