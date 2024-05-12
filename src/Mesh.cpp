#include "Mesh.h"



/*
* Class functions
*/

void Mesh::generateNormals() {
    FaceNormals.resize(Faces.size());
    EdgeMap.resize(Faces.size() * 2);

    for (size_t i = 0; i < Faces.size(); ++i) {
        const glm::vec3 PointA = Indicies[Faces[i].x].Pos;
        const glm::vec3 PointB = Indicies[Faces[i].y].Pos;
        const glm::vec3 PointC = Indicies[Faces[i].z].Pos;

        const glm::vec3 ray_one = PointB - PointA;
        const glm::vec3 ray_two = PointC - PointA;

        FaceNormals[i] = glm::normalize(glm::cross(ray_one, ray_two));

        EdgeMap[i * 2] = ray_one;
        EdgeMap[(i * 2) + 1] = ray_two;
    }
}


Ray Mesh::generateRandomVecOnFace(const size_t faceIdx, const glm::vec3 & origin) const {
    glm::vec3 randVec = generateRandomVecH();
    glm::vec3 normal = getFaceNormal(faceIdx);
    randVec *= glm::dot(randVec, normal) < 0?-1:1;

    glm::vec3 newOrigin = origin + (randVec * .001f); //avoid shadow acne
    return Ray(newOrigin, randVec);
}


Ray Mesh::generateLambertianVecOnFace(const size_t faceIdx, const glm::vec3 & origin) const {
    glm::vec3 newDir = getFaceNormal(faceIdx) + generateRandomVecH();
    glm::vec3 newOrigin = origin + (newDir * .001f); // avoid shadow acne
    return Ray(newOrigin, newDir);
}




/*
* Getters
*/
Color Mesh::getColor(size_t face_idx, float u, float v, float w) const {

    // Texture Coordinates for all of the above.
    // W * a, U * B, C * v.
    const glm::vec2 & PointA = Indicies[Faces[face_idx].x].TQ;
    const glm::vec2 & PointB = Indicies[Faces[face_idx].y].TQ;
    const glm::vec2 & PointC = Indicies[Faces[face_idx].z].TQ;

    float x_average = w * PointA.x + u * PointB.x + v * PointC.x;
    float y_average = w * PointA.y + u * PointB.y + v * PointC.y;
    
    return mat.getDiffuse(); // TODO : CHANGE
}

Material const & Mesh::getMaterial() const {
    return mat;
}


const glm::vec3 & Mesh::getFaceNormal(size_t idx) const {
    return FaceNormals[idx];
}


/*
* Setters
*/
void Mesh::setColor(float r, float g, float b) {
    mat.setDiffuse(Color(r, g, b));
}



/*
* Helpful debugging functions
*/

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

        std::cout << "Direction : ";

        std::cout << mesh.FaceNormals[i].x << " ";
        std::cout << mesh.FaceNormals[i].y << " ";
        std::cout << mesh.FaceNormals[i].z << "\n";
    }
}

void printMesh(const Mesh & mesh) {
    printMeshFaces(mesh);
    std::cout << "\n\n\n";
    printMeshNormals(mesh);
    std::cout << "\n\n\n";
    printMeshVertexes(mesh);
}


