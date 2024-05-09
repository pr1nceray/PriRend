#pragma once

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include <vector>
#include <iostream>
#include <string>

struct Vertex
{
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;

};


struct Mesh
{
    //note : Contains duplicates and not in order of obj file.
    //need to find a way to trim down on vertexes maybe.
    std::vector<Vertex> Indicies;
    std::vector<glm::vec3> Faces;

    std::vector<glm::vec3> FaceNormals;
    std::vector<glm::vec3> FaceNormalOrigins; //potentially uneeded. if unused, delete.
    std::vector<glm::vec3> EdgeMap;

    uint32_t mat_index;
    Mesh() : mat_index(0)
    {
    }

    Mesh(const std::vector<Vertex> & Indicies_in) :
    Indicies(Indicies_in), mat_index(0)
    {
    }

    void generateNormals()
    {
        FaceNormals.resize(Faces.size());
        FaceNormalOrigins.resize(Faces.size());
        EdgeMap.resize(Faces.size() * 2);

        for(size_t i = 0; i < Faces.size(); ++i)
        {
        
        const glm::vec3 PointA = Indicies[Faces[i].x].Pos;
        const glm::vec3 PointB = Indicies[Faces[i].y].Pos;
        const glm::vec3 PointC = Indicies[Faces[i].z].Pos;

        const glm::vec3 ray_one = glm::vec3(PointB.x - PointA.x, PointB.y - PointA.y, PointB.z - PointA.z);
        const glm::vec3 ray_two = glm::vec3(PointC.x - PointA.x, PointC.y - PointA.y, PointC.z - PointA.z);

        float x_sum = PointA.x + PointB.x + PointC.x;
        float y_sum = PointA.y + PointB.y + PointC.y;
        float z_sum = PointA.z + PointB.z + PointC.z;
    
        FaceNormals[i] = glm::normalize(glm::cross(ray_one,ray_two));
        FaceNormalOrigins[i] = glm::vec3(x_sum/3.0, y_sum/3.0, z_sum/3.0);

        EdgeMap[i * 2] = ray_one;
        EdgeMap[(i * 2) + 1] = ray_two;
        
        }
    }
    
};


//should move to the actual struct

void printMeshVertexes(const Mesh & mesh)
{
    std::cout << "Printing Vertexes : \n";
    for(size_t i = 0; i < mesh.Indicies.size(); ++i)
    {
        std::cout << mesh.Indicies[i].Pos.x << " ";
        std::cout << mesh.Indicies[i].Pos.y << " ";
        std::cout << mesh.Indicies[i].Pos.z << "\n";
    }
}

void printMeshFaces(const Mesh & mesh)
{
    std::cout << "Printing Faces : \n";
    for(size_t i = 0; i < mesh.Faces.size(); ++i)
    {
        std::cout << mesh.Faces[i].x << " ";
        std::cout << mesh.Faces[i].y << " ";
        std::cout << mesh.Faces[i].z << "\n";
    }
}

void printMeshNormals(const Mesh & mesh)
{
    std::cout << "Printing Normals : \n";
    for(size_t i = 0; i < mesh.FaceNormals.size(); ++i)
    {
        std::cout << "Origin : ";

        std::cout << mesh.FaceNormalOrigins[i].x << " ";
        std::cout << mesh.FaceNormalOrigins[i].y << " ";
        std::cout << mesh.FaceNormalOrigins[i].z << " ";

        std::cout << "Direction : ";

        std::cout << mesh.FaceNormals[i].x << " ";
        std::cout << mesh.FaceNormals[i].y << " ";
        std::cout << mesh.FaceNormals[i].z << "\n";
    }
}

void printMesh(const Mesh & mesh)
{
    printMeshVertexes(mesh);
    std::cout << "\n\n\n";
    printMeshFaces(mesh);
    std::cout << "\n\n\n";
    printMeshNormals(mesh);
}