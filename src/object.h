#pragma once
#include "Vertex.h"
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

class object
{
    public:
    object(const std::string & file_name)
    {
        Assimp::Importer importer;
        const aiScene * scene = importer.ReadFile(file_name, aiProcess_Triangulate); //can add more cooler options.
        if(!scene || !scene->mRootNode)
        {
            throw std::runtime_error("Error loading object");
        }
        
        CreateMeshes(scene->mRootNode, scene);

    }

    /*
    * Getters
    */

    const std::vector<Mesh> & getObjInfo() const
    {
        return objInfo;
    }

    private:
    glm::vec3 Center;
    glm::vec3 Rot;
    std::vector<Mesh> objInfo;

    void CreateMeshes(aiNode * node, const aiScene * scene)
    {
        for(size_t i = 0; i < node->mNumMeshes;++i)
        {
            aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];   
            objInfo.push_back(processMesh(mesh, scene));
        }

        for(size_t i = 0; i < node->mNumChildren;++i)
        {
            CreateMeshes(node->mChildren[i],scene);
        }
    }

    Mesh processMesh(aiMesh * mesh, const aiScene * scene)
    {
        Mesh meshlcl;

        for(size_t i = 0; i < mesh->mNumVertices;++i)
        {
            //y,z flipped in obj format
            //y in obj is also mirrored (so we mulitply z by -1, since z is rlly y)!
            glm::vec3 pos(mesh->mVertices[i].x, -1 * mesh->mVertices[i].z, mesh->mVertices[i].y);
            glm::vec3 norm(mesh->mNormals[i].x, -1 * mesh->mNormals[i].z, mesh->mNormals[i].y);
            glm::vec2 TQ;

            if(mesh->mTextureCoords[0])
            {
                TQ = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            }
            else{
                TQ = glm::vec2(0,0);
            }
            Vertex v_add {pos, norm, TQ};
            meshlcl.Indicies.push_back(v_add);
        }

        //std::cout << "num vertexes : " << meshlcl.Indicies.size() << "\n";
        for(size_t i = 0; i < mesh->mNumFaces; ++i)
        {
            aiFace face_add = mesh->mFaces[i];
            glm::vec3 index_for_face;
            
            //Gaurenteed <= 3 vertexes due to assimp option above.
            index_for_face.x = face_add.mIndices[0];
            index_for_face.y = face_add.mIndices[1];
            index_for_face.z = face_add.mIndices[2];

            meshlcl.Faces.push_back(index_for_face);
        }
        
        meshlcl.generateNormals();
        printMesh(meshlcl);
        return meshlcl;
    }

};