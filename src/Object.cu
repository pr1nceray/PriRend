#include "Object.cuh"

Object::Object(const std::string & fileName, std::vector<Material> & mats) {
    Assimp::Importer importer;

    // Can add more cooler options.
    const aiScene * scene = importer.ReadFile(fileName, aiProcess_Triangulate);

    if (!scene || !scene->mRootNode) {
        throw std::runtime_error("Error loading object");
    }

    CreateMeshes(scene->mRootNode, scene, mats.size());
    CreateMaterials(scene, mats);
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


/*
* Internal functions for setting up the object
*/

void Object::CreateMeshes(aiNode * node, const aiScene * scene, size_t baseMatIdx) {
    if(!scene->HasMeshes()) {
        return;
    }

    for (size_t i = 0; i < node->mNumMeshes; ++i) {
        aiMesh * mesh = scene->mMeshes[node->mMeshes[i]];
        objInfo.push_back(processMesh(mesh, scene, baseMatIdx));
    }

    for (size_t i = 0; i < node->mNumChildren; ++i) {
        CreateMeshes(node->mChildren[i], scene, baseMatIdx);
    }
}

Mesh Object::processMesh(aiMesh * mesh, const aiScene * scene, size_t baseMatIdx) {
    Mesh meshlcl;
    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
        
        //DOES NOT WORK FOR OBJ FILES
        // SEE BELOW
        // y,z flipped in obj format
        // y in obj is mirrored, so we multiply by -1
        // ALSO CHECK PREVIOUS GIT HISTORY FOR MORE INFO
        glm::vec3 pos(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        glm::vec3 norm(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        glm::vec2 TQ = glm::vec2(0.0f, 0.0f);

        //only accept the first vertex coord.
        if (mesh->mTextureCoords[0]) {
            TQ = glm::vec2(mesh->mTextureCoords[0][i].x,
            mesh->mTextureCoords[0][i].y);
        }
        meshlcl.Indicies.push_back(Vertex{pos, norm, TQ});
    }

    meshlcl.MaterialIdx = baseMatIdx + static_cast<size_t>(mesh->mMaterialIndex); 

    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
        aiFace face_add = mesh->mFaces[i];
        glm::ivec3 index_for_face;

        index_for_face.x = face_add.mIndices[0];
        index_for_face.y = face_add.mIndices[1];
        index_for_face.z = face_add.mIndices[2];
        meshlcl.Faces.push_back(index_for_face);
    }

    meshlcl.generateNormals();
    printMesh(meshlcl);
    return meshlcl;
}

void Object::CreateMaterials(const aiScene * scene, std::vector<Material> & materials) {
    if(!scene->HasMaterials()) {
        return;
    }

    for (size_t i = 0; i < scene->mNumMaterials; ++ i) {
        processMaterials(scene->mMaterials[i], materials);
    }
}

void Object::processMaterials(const aiMaterial * mat, std::vector<Material> & materials) {
    Material matToAdd;
    Color white = Color(1.0f, 1.0f, 1.0f);
    Color roughness = Color(.5f, .5f, .5f);
    Color black = Color();
    checkBasic(&matToAdd, mat, aiTextureType_DIFFUSE, white);
    checkBasic(&matToAdd, mat, aiTextureType_NORMALS, black);
    checkBasic(&matToAdd, mat, aiTextureType_SPECULAR, black);
    checkBasic(&matToAdd, mat, aiTextureType_METALNESS, black);
    checkBasic(&matToAdd, mat, aiTextureType_DIFFUSE_ROUGHNESS, black);

}
void Object::checkBasic(Material * mat, const aiMaterial * matptr, aiTextureType type, Color c) {
    if(matptr->GetTextureCount(type) == 0) {
        mat->setBasic(c, type);
    }
    processTextures(mat, matptr, type);
}
void Object::processTextures(Material * matToAdd, const aiMaterial * matptr, aiTextureType type) {
    for(size_t i = 0; i < matptr->GetTextureCount(type); ++i){
        if(i >= 1) {
            throw std::runtime_error("Unable to process current object; Material has more than 1 of the same kind of texture");
        }
        aiString str;
        matptr->GetTexture(aiTextureType_DIFFUSE, i, &str);
        matToAdd->loadTexture(str.C_Str(), type);
    }
}