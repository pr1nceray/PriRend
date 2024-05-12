#pragma once
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <string>
#include <assimp/Importer.hpp>

#include "./Mesh.cuh"
#include "./Primitives.cuh"

class Object {
    public:
    explicit Object(const std::string & file_name);

    Color getMeshColor(const CollisionInfo & info) const {
        float w = 1.0f - info.CollisionPoint.x - info.CollisionPoint.y;
        return objInfo[info.meshIdx].getColor(info.faceIdx, w, info.CollisionPoint.x, info.CollisionPoint.y);
    }

    const glm::vec3 & getFaceNormal(const size_t meshIdx, const size_t faceIdx) const {
        return objInfo[meshIdx].getFaceNormal(faceIdx);
    }


    void setMeshColors(float r, float g, float b);
    
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
