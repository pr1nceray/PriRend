#pragma once
#include <time.h>
#include <vector>
#include <iostream>
#include <string>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "./Color.h"
#include "./Primitives.h"


struct Mesh {
    // note : Contains duplicates and not in order of obj file.
    // need to find a way to trim down on vertexes maybe.
    std::vector<Vertex> Indicies;
    std::vector<glm::vec3> Faces;

    std::vector<glm::vec3> FaceNormals;
    std::vector<glm::vec3> FaceNormalOrigins;  // potentially uneeded.
    std::vector<glm::vec3> EdgeMap;

    Color mat;
    uint32_t mat_index;
    Mesh() : mat_index(0) {
        uint8_t color = generateRandomNum();
        mat = Color(color, color, color);
    }

    explicit Mesh(const std::vector<Vertex> & Indicies_in) :
    Indicies(Indicies_in), mat_index(0) {
        uint8_t color = generateRandomNum();
        mat = Color(color, color, color);
    }

    void generateNormals();

    /*
    * Getters
    */

    const glm::vec3 & getFaceNormal(size_t idx) const;

    const Color & getColor(size_t face_idx, float u, float v, float w) const;
};

void printMeshVertexes(const Mesh & mesh);

void printMeshFaces(const Mesh & mesh);

void printMeshNormals(const Mesh & mesh);

void printMesh(const Mesh & mesh);
