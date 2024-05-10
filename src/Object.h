#pragma once
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <string>
#include <assimp/Importer.hpp>

#include "./Mesh.h"

class Object {
    public:
    explicit Object(const std::string & file_name);

    /*
    * Getters
    */

    const std::vector<Mesh> & getObjInfo() const;
    const glm::vec3 & getRot() const;
    const glm::vec3 & getCenter() const;

    private:
    glm::vec3 Center;
    glm::vec3 Rot;
    std::vector<Mesh> objInfo;

    /*
    * Internal functions for setting up the object
    */

    void CreateMeshes(aiNode * node, const aiScene * scene);

    Mesh processMesh(aiMesh * mesh, const aiScene * scene);
};
