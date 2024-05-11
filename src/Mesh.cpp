#include "Mesh.h"


Color Mesh::getColor(size_t face_idx, float u, float v, float w) const {

    //Texture Coordinates for all of the above.
    //W * a, U * B, C * v.
    const glm::vec2 & PointA = Indicies[Faces[face_idx].x].TQ;
    const glm::vec2 & PointB = Indicies[Faces[face_idx].y].TQ;
    const glm::vec2 & PointC = Indicies[Faces[face_idx].z].TQ;

    float x_average = w * PointA.x + u * PointB.x + v * PointC.x;
    float y_average = w * PointA.y + u * PointB.y + v * PointC.y;
    
    return mat.getDiffuse();
}


const Color Mesh::getTextureAt(float x, float y) const {
    return Color(1.0f, 1.0f, 1.0f); //mat;
}

const glm::vec3 & Mesh::getFaceNormal(size_t idx) const {
    return FaceNormals[idx];
}

void Mesh::setColor(float r, float g, float b) {
    mat.setDiffuse(Color(r, g, b));
}


Material const & Mesh::getMaterial() const {
    return mat;
}

void Mesh::generateNormals() {
    FaceNormals.resize(Faces.size());
    FaceNormalOrigins.resize(Faces.size());
    EdgeMap.resize(Faces.size() * 2);

    for (size_t i = 0; i < Faces.size(); ++i) {
        const glm::vec3 PointA = Indicies[Faces[i].x].Pos;
        const glm::vec3 PointB = Indicies[Faces[i].y].Pos;
        const glm::vec3 PointC = Indicies[Faces[i].z].Pos;

        const glm::vec3 ray_one = PointB - PointA;
        const glm::vec3 ray_two = PointC - PointA;

        float x_sum = PointA.x + PointB.x + PointC.x;
        float y_sum = PointA.y + PointB.y + PointC.y;
        float z_sum = PointA.z + PointB.z + PointC.z;

        FaceNormals[i] = glm::normalize(glm::cross(ray_one, ray_two));
        FaceNormalOrigins[i] = glm::vec3(x_sum/3.0, y_sum/3.0, z_sum/3.0);

        EdgeMap[i * 2] = ray_one;
        EdgeMap[(i * 2) + 1] = ray_two;
    }
}


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


