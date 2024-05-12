#include "Object.h"

Object::Object(const std::string & filename) {
    Assimp::Importer importer;

    // Can add more cooler options.
    const aiScene * scene = importer.ReadFile(filename, aiProcess_Triangulate);

    if (!scene || !scene->mRootNode) {
        throw std::runtime_error("Error loading object");
    }

    CreateMeshes(scene->mRootNode, scene);
}

/*
* Getters for the Object class.
*/

const std::vector<Mesh> & Object::getObjInfo() const {
    return objInfo;
}

const glm::vec3 & Object::getRot() const {
    return Rot;
}

const glm::vec3 & Object::getCenter() const {
    return Center;
}

void Object::setMeshColors(float r, float g, float b) {
    for(size_t i = 0; i < objInfo.size(); ++i) {
        objInfo[i].setColor(r, g, b);
    }
}

/*
* Internal functions for setting up the object
*/

void Object::CreateMeshes(aiNode * node, const aiScene * scene) {
    for (size_t i = 0; i < node->mNumMeshes; ++i) {
        aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
        objInfo.push_back(processMesh(mesh, scene));
    }

    for (size_t i = 0; i < node->mNumChildren; ++i) {
        CreateMeshes(node->mChildren[i], scene);
    }
}

Mesh Object::processMesh(aiMesh * mesh, const aiScene * scene) {
        Mesh meshlcl;
        for (size_t i = 0; i < mesh->mNumVertices; ++i) {
            // y,z flipped in obj format
            // y in obj is mirrored, so we multiply by -1
            glm::vec3 pos(mesh->mVertices[i].x,
                -1 * mesh->mVertices[i].z, mesh->mVertices[i].y);
            glm::vec3 norm(mesh->mNormals[i].x,
                -1 * mesh->mNormals[i].z, mesh->mNormals[i].y);

            glm::vec2 TQ = glm::vec2(0, 0);

            if (mesh->mTextureCoords[0]) {
                TQ = glm::vec2(mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y);
            }

            Vertex v_add {pos, norm, TQ};
            meshlcl.Indicies.push_back(v_add);
        }

        for (size_t i = 0; i < mesh->mNumFaces; ++i) {
            aiFace face_add = mesh->mFaces[i];
            glm::vec3 index_for_face;

            // Gaurenteed <= 3 vertexes due to assimp option above.
            index_for_face.x = face_add.mIndices[0];
            index_for_face.y = face_add.mIndices[1];
            index_for_face.z = face_add.mIndices[2];

            meshlcl.Faces.push_back(index_for_face);
        }

        meshlcl.generateNormals();
        printMesh(meshlcl);
        return meshlcl;
}
