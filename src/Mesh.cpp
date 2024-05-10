#include "Mesh.h"

void printMeshVertexes(const Mesh & mesh) {
    std::cout << "Printing Vertexes : \n";
    for (size_t i = 0; i < mesh.Indicies.size(); ++i) {
        std::cout << mesh.Indicies[i].Pos.x << " ";
        std::cout << mesh.Indicies[i].Pos.y << " ";
        std::cout << mesh.Indicies[i].Pos.z << "\n";
    }
}

void printMeshFaces(const Mesh & mesh) {
    std::cout << "Printing Faces : \n";
    for (size_t i = 0; i < mesh.Faces.size(); ++i) {
        std::cout << mesh.Faces[i].x << " ";
        std::cout << mesh.Faces[i].y << " ";
        std::cout << mesh.Faces[i].z << "\n";
    }
}

void printMeshNormals(const Mesh & mesh) {
    std::cout << "Printing Normals : \n";
    for (size_t i = 0; i < mesh.FaceNormals.size(); ++i) {
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

void printMesh(const Mesh & mesh) {
    printMeshVertexes(mesh);
    std::cout << "\n\n\n";
    printMeshFaces(mesh);
    std::cout << "\n\n\n";
    printMeshNormals(mesh);
}