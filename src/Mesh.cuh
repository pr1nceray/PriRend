#pragma once
#include <time.h>
#include <vector>
#include <iostream>
#include <string>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.cuh"
#include "./Primitives.cuh"
#include "./Materials.cuh"

struct Mesh {
    // note : Contains duplicates and not in order of obj file.
    // need to find a way to trim down on vertexes maybe.
    std::vector<Vertex> Indicies;
    std::vector<glm::ivec3> Faces; //potentially uneeded since only used for normal creation
    std::vector<glm::vec3> FaceNormals;
    std::vector<glm::vec3> EdgeMap;

    Material mat;

    Mesh() {
        mat.setDiffuse(Color(generateRandomFloatH(), generateRandomFloatH(), generateRandomFloatH()));
    }

    explicit Mesh(const std::vector<Vertex> & Indicies_in) :
    Indicies(Indicies_in) {
        mat.setDiffuse(Color(generateRandomFloatH(), generateRandomFloatH(), generateRandomFloatH()));
    }


    /*
    * Class functions
    */
    void generateNormals();

    Ray generateRandomVecOnFace(const size_t faceIdx, const glm::vec3 & origin) const;
    Ray generateLambertianVecOnFace(const size_t faceIdx, const glm::vec3 & origin) const;

    /*
    * Getters
    */
    Material const & getMaterial() const;
    Color getColor(size_t face_idx, float u, float v, float w) const;
    const glm::vec3 & getFaceNormal(size_t idx) const;

    /*
    * Setters
    */
    void setColor(float r, float g, float b);
    
};

/*
* Functions for debug output.
*/

void printMesh(const Mesh & mesh);
