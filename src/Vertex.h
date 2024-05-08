#pragma once
#include <string>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <vector>
#include <iostream>

struct Vertex
{
    glm::vec3 Pos;
    glm::vec3 Normal;
    glm::vec2 TQ;

};

struct Mesh
{
    std::vector<Vertex> Indicies;
    std::vector<glm::vec3> Faces;
    uint32_t mat_index;
    Mesh() : mat_index(0)
    {
    }

    Mesh(const std::vector<Vertex> & Indicies_in) :
    Indicies(Indicies_in), mat_index(0)
    {
    }
};

void printMesh(const Mesh & mesh)
{
    for(size_t i = 0; i < mesh.Indicies.size(); ++i)
    {
        std::cout << mesh.Indicies[i].Pos.x << " ";
        std::cout << mesh.Indicies[i].Pos.y << " ";
        std::cout << mesh.Indicies[i].Pos.z << "\n";

        //std::cout << mesh.Indicies[i].Normal.x << " ";
        //std::cout << mesh.Indicies[i].Normal.y << " ";
        //std::cout << mesh.Indicies[i].Normal.z << "\n";

        //std::cout << mesh.Indicies[i].TQ.x << " ";
        //std::cout << mesh.Indicies[i].TQ.y << "\n \n";
    }

    for(size_t i = 0; i < mesh.Faces.size(); ++i)
    {
        std::cout << mesh.Faces[i].x << " ";
        std::cout << mesh.Faces[i].y << " ";
        std::cout << mesh.Faces[i].z << "\n" << std::endl;
    }

}