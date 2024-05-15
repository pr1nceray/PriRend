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

void Object::setMeshColors(float r, float g, float b) {
    for(size_t i = 0; i < objInfo.size(); ++i) {

    }
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
        if (mesh->mTextureCoords[0]) {
            TQ = glm::vec2(mesh->mTextureCoords[0][i].x,
            mesh->mTextureCoords[0][i].y);
        }

        meshlcl.matIdx = baseMatIdx + static_cast<size_t>(mesh->mMaterialIndex); 
        
        meshlcl.Indicies.push_back(Vertex{pos, norm, TQ});
    }

    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
        aiFace face_add = mesh->mFaces[i];
        glm::ivec3 index_for_face;
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

void Object::CreateMaterials(const aiScene * scene, std::vector<Material> & materials) {
    if(!scene->HasMaterials()) {
        return;
    }

    for (size_t i = 0; i < scene->mNumMaterials; ++ i) {
        processDiffuse(scene->mMaterials[i], materials);
    }
}

void Object::processDiffuse(const aiMaterial * mat, std::vector<Material> & materials) {
    Material matToAdd;
    for(size_t i = 0; i < mat->GetTextureCount(aiTextureType_DIFFUSE); ++i){
        if(i >= 1 ) {
            throw std::runtime_error("Unable to process current object; Material has more than 1 diffuse texture");
        }
        aiString str;
        mat->GetTexture(aiTextureType_DIFFUSE, i, &str);
        matToAdd.loadDiffuse(str.C_Str());
    }
    materials.push_back(matToAdd);
}