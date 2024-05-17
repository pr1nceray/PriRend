#include "./GpuInfo.cuh"

__device__ GpuInfo * sceneInfo;

/*
* TODO : Materials
*/
/*
__device__ Material const & MeshGpu::getMaterial() const {

}
*/

__device__ const glm::vec3 & MeshGpu::getFaceNormal(size_t idx) const {
    return normalBuff[idx];
}

size_t GpuInfo::sumMeshSizes(const std::vector<Mesh> & meshIn) const {
    size_t total = 0;
    for (size_t i = 0; i < meshIn.size(); ++i) {
        total += sizeof(glm::vec3) * meshIn[i].FaceNormals.size();
        total += sizeof(glm::vec3) * meshIn[i].EdgeMap.size();
        total += sizeof(glm::ivec3) * meshIn[i].Faces.size();
        total += sizeof(Vertex) * meshIn[i].Indicies.size(); 
    }
    return total;
}

void GpuInfo::copyIntoDevice(const std::vector<Mesh> & meshIn, const std::vector<Material> & matIn) {

    copyMeshData(meshIn);
    copyMaterialData(matIn);
    //copy over all the info into global var
    GpuInfo * tmp;
    // malloc the gpu info struct
    handleCudaError(cudaMalloc((void **)&tmp, sizeof(GpuInfo)));
    printMeshInfo<<<1,1,1>>>(tmp);
    //copy over the gpu info struct
    handleCudaError(cudaMemcpy((void *) tmp, this, sizeof(GpuInfo), cudaMemcpyHostToDevice));
    //copy over the pointer to the gpu struct.  
    handleCudaError(cudaMemcpyToSymbol(sceneInfo, &tmp, sizeof(GpuInfo *)));
}

void GpuInfo::copyMeshData(const std::vector<Mesh> & meshIn) {
    meshLen = meshIn.size();

    size_t sizeOfMeshes = sumMeshSizes(meshIn); //size of the information
    cudaError_t err = cudaMalloc((void **)(&infoBuffer), sizeOfMeshes); //malloc the info
    handleCudaError(err);

    size_t sizeOfArray = sumMeshArr(meshIn); // size of the struct to hold the information

    err = cudaMalloc((void **)(&meshDev), sizeOfArray); //malloc the array of ptrs
    handleCudaError(err);

    MeshGpu * meshHost = new MeshGpu[sizeOfArray]; //create information holder on host

    void * bufferCpy = infoBuffer;
    copyNormalBuff(bufferCpy, meshIn, meshHost);
    copyEdgeBuff(bufferCpy, meshIn, meshHost);
    copyFaceBuff(bufferCpy, meshIn, meshHost);
    copyVertexBuff(bufferCpy, meshIn, meshHost);
    copyMaterialIndex(meshIn, meshHost);
    setLengthMesh(meshIn, meshHost);
    handleCudaError(cudaMemcpy(meshDev, meshHost, sizeOfArray, cudaMemcpyHostToDevice));

    delete[] meshHost; //no longer needed, free resources.
}

void GpuInfo::copyMaterialData(const std::vector<Material> & matIn) {
    
    /*
    * Obtain material map
    * malloc the size of the texture info
    * then, malloc the size of the images
    * Copy over images
    */
    auto & matMap = Material::getTextures();
    
    size_t textureBuffSize = sumTextArr(matMap);
    size_t textureInfoSize = sumTextInfoSize(matMap);
    size_t materialBuffSize = sumMatArr(matIn);
    std::unordered_map<uintptr_t, TextInfo *> textureTranslate;
    handleCudaError(cudaMalloc((void **)&textureBuffer, textureBuffSize)); //malloc the raw texture data buffer
    handleCudaError(cudaMalloc((void **)&textureInfo, textureInfoSize)); //malloc the other info (ptr to data, size, width)
    handleCudaError(cudaMalloc((void **)&matDev, materialBuffSize));

    //copy over the buffers
    MatGpu * matHost = new MatGpu[matIn.size()];
    TextInfo * textHost = new TextInfo[matMap.size()];
    void * textStart = textureBuffer;

    // NOTE : slow due the amount of calls to cudaMemcpy
    // possible improvement : use a vector of floats instead of map
    // and have one big cudaMemcpy call?

    size_t idx = 0;
    for(auto it : matMap) {
        // copy over all textures into texture buffer
        size_t sizeCpy = CHANNEL * it.second->width * it.second->height;
        handleCudaError(cudaMemcpy(textStart, it.second->arr, sizeof(float) * sizeCpy, cudaMemcpyHostToDevice));
        textHost[idx] = *it.second;
        textHost[idx].arr = static_cast<float *>(textStart); // set array to point to somewhere in raw texture buffer
        textureTranslate[reinterpret_cast<uintptr_t>(it.second)] = (textureInfo + idx); 
        textStart = static_cast<void *>(static_cast<float *>(textStart) + sizeCpy); // increment buffer
        idx++;
    }
    // copy over textureInfo buffer
    handleCudaError(cudaMemcpy((void *)textureInfo, textHost, matMap.size() * sizeof(TextInfo), cudaMemcpyHostToDevice));

    textLen = matMap.size();
    for (size_t i = 0; i < matIn.size(); i++) {
        matHost[i].diffuse = checkTextureInMap(reinterpret_cast<uintptr_t>(matIn[i].getDiffuse()), textureTranslate);
        matHost[i].normals = checkTextureInMap(reinterpret_cast<uintptr_t>(matIn[i].getNormal()), textureTranslate);
        matHost[i].specular = checkTextureInMap(reinterpret_cast<uintptr_t>(matIn[i].getSpecular()), textureTranslate);
        matHost[i].metallic = checkTextureInMap(reinterpret_cast<uintptr_t>(matIn[i].getMetallic()), textureTranslate);
        matHost[i].roughness = checkTextureInMap(reinterpret_cast<uintptr_t>(matIn[i].getRoughness()), textureTranslate);
    }
    handleCudaError(cudaMemcpy(matDev, matHost, matIn.size() * sizeof(MatGpu), cudaMemcpyHostToDevice));
    matLen = matIn.size();
    //free what we no longer need
    delete[] matHost;
    delete[] textHost;
}

TextInfo * GpuInfo::checkTextureInMap(uintptr_t txthost, const std::unordered_map<uintptr_t, TextInfo *> & textures) {
    auto it = textures.find(txthost);
    if (it == textures.end()) {
            throw std::runtime_error("A material has a texture that we never loaded");
    }
    return it->second;
}
/*
* Since there should only be ONE Gpu info class at a time, 
* and we are destructing, then we should free the resources that
* we allocated in copyIntoDevice.
*/
 __host__ void GpuInfo::freeResources() {
    cudaFree(infoBuffer);
    cudaFree(meshDev);
    cudaFree(matDev);
    cudaFree(textureBuffer);
    cudaFree(textureInfo);
    cudaFree(sceneInfo);
 }


/*
* Copy over all the vertex buffers one after each other for cache purpouses.
* Buffer copy is so that we dont lose infoBuffer.
* take the size of the normalVector array, and copy over the data into where
* BufferCopy is pointing. From there, set the pointer of mesh's to be accurate.
* Then add the size that we copied.
* same for TQ and edgemap.
*/

template<typename T>
void GpuInfo::copyBuff(void * & start, const std::vector<T> * data, T * & write) {
    cudaError_t err;
    size_t sizeOfBuff = sizeof(T) * data->size();
    err = cudaMemcpy(start, data->data(), sizeOfBuff, cudaMemcpyHostToDevice);
    handleCudaError(err);
    write = static_cast<T *>(start);
    start = (void *)(static_cast<T *>(start) + data->size());
}

void GpuInfo::copyNormalBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for (size_t i = 0; i < meshIn.size(); ++i) { 
        copyBuff<glm::vec3>(start, &meshIn[i].FaceNormals, meshHost[i].normalBuff);
    }
}
void GpuInfo::copyEdgeBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for (size_t i = 0; i < meshIn.size(); ++i) { 
        copyBuff<glm::vec3>(start, &meshIn[i].EdgeMap, meshHost[i].edgeBuff);
    }

}
void GpuInfo::copyFaceBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for (size_t i = 0; i < meshIn.size(); ++i) { 
        copyBuff<glm::ivec3>(start, &meshIn[i].Faces, meshHost[i].faceBuff);
    }
}
void GpuInfo::copyVertexBuff(void * & start, const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for (size_t i = 0; i < meshIn.size(); ++i) { 
        copyBuff<Vertex>(start, &meshIn[i].Indicies, meshHost[i].vertexBuffer);
    }
}

void GpuInfo::copyMaterialIndex(const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for (size_t i = 0; i < meshIn.size(); ++i) {
        //meshHost[i].matIdx = meshIn[i].diffIdx;
    }
}
/*
* Set the lengths for the fields inside of infobuffer
*/
void GpuInfo::setLengthMesh( const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for(size_t i = 0; i < meshIn.size(); ++i) {
        meshHost[i].faceSize = meshIn[i].Faces.size();
        meshHost[i].vertexSize = meshIn[i].Indicies.size();
    }
}

/*
* The size that the array of structural data takes up
*/
size_t GpuInfo::sumMeshArr(const std::vector<Mesh> & meshIn) const {
    return sizeof(MeshGpu) * meshIn.size();
}

/*
* Size needed to allocate all the textures and their respective textInfo
* Should be the size of the textInfo buffer and the size of the respective textures
*/
size_t GpuInfo::sumTextArr(const std::unordered_map<std::string, TextInfo *> & textures) const {
    size_t numElem = 0;
    for (auto it : textures) {
        numElem += (sizeof(float) * (it.second)->height * (it.second)->width * CHANNEL);
    }
    return numElem;
   
}

size_t GpuInfo::sumTextInfoSize(const std::unordered_map<std::string, TextInfo *> & textures) const {
    return sizeof(TextInfo) * textures.size();
}
/*
* Size of all the Materials that is needed
*/
size_t GpuInfo::sumMatArr(const std::vector<Material> & matIn) const {
    return sizeof(MatGpu) * matIn.size();
}


/*
* Useful functions for debugging
*/
__device__ void printMeshFaces(MeshGpu * mesh) {
    printf("Printing Faces\n");
    for (size_t j = 0; j < mesh->faceSize; ++j) {
        printf("%d  %d  %d \n",
        mesh->faceBuff[j].x, 
        mesh->faceBuff[j].y, 
        mesh->faceBuff[j].z);
    }
}

__device__ void printMeshNormals(MeshGpu * mesh) {
    printf("Printing Face Normals\n");
    for (size_t j = 0; j < mesh->faceSize; ++j) {
        printf("%.6f  %.6f  %.6f \n",
        mesh->normalBuff[j].x, 
        mesh->normalBuff[j].y, 
        mesh->normalBuff[j].z);
    }
}

__device__ void printMeshEdges(MeshGpu * mesh) {
  printf("Printing Edges\n");
    for (size_t j = 0; j < mesh->faceSize; ++j) {
        printf("%.6f  %.6f  %.6f ",
        mesh->edgeBuff[j * 2].x, 
        mesh->edgeBuff[j * 2].y, 
        mesh->edgeBuff[j * 2].z);

        printf("%.6f  %.6f  %.6f \n",
        mesh->edgeBuff[(j * 2) + 1].x, 
        mesh->edgeBuff[(j * 2) + 1].y, 
        mesh->edgeBuff[(j * 2) + 1].z);

    }
}

__device__ void printMeshVertexs(MeshGpu * mesh) {
    printf("Printing Vertexes\n");
    for (size_t j = 0; j < mesh->vertexSize; ++j) {
        printf(" %.6f %.6f %.6f \n",
        mesh->vertexBuffer[j].Pos.x, 
        mesh->vertexBuffer[j].Pos.y, 
        mesh->vertexBuffer[j].Pos.z);
    }
}



__global__ void printMeshInfo(GpuInfo inf) {
    for (size_t i = 0; i < inf.meshLen; ++i) {
        printf("Printing for mesh %d.\n", static_cast<int>(i));
        MeshGpu cur = inf.meshDev[i];
        printMeshFaces(&cur);
        printMeshNormals(&cur);
        printMeshEdges(&cur);
        printMeshVertexs(&cur);
    }
}
__global__ void printMeshInfo(GpuInfo * inf) {
    for (size_t i = 0; i < inf->meshLen; ++i) {
        printf("Printing for mesh %d.\n", static_cast<int>(i));
        MeshGpu cur = inf->meshDev[i];
        printMeshFaces(&cur);
        printMeshNormals(&cur);
        printMeshEdges(&cur);
        printMeshVertexs(&cur);
        printf("Material idx : %d", cur.matIdx);
    }
}

__global__ void printMeshGlobal() {

    for (size_t i = 0; i < sceneInfo->meshLen; ++i) {
        printf("Printing for mesh %d.\n", static_cast<int>(i));
        MeshGpu cur = sceneInfo->meshDev[i];
        printMeshFaces(&cur);
        printMeshNormals(&cur);
        printMeshEdges(&cur);
        printMeshVertexs(&cur);
    }
}

__global__ void printMaterialInfo() {

    for (size_t i = 0; i < sceneInfo->matLen; ++i) {
        printf("Printing for material : %d \n", static_cast<int>(i));
        printf("Width : %d Height : %d Texture Address : %p \n",
        sceneInfo->matDev[i].diffuse->width, 
        sceneInfo->matDev[i].diffuse->height, 
        sceneInfo->matDev[i].diffuse->arr);
        printTextures(sceneInfo->matDev[i].diffuse);
    }
}