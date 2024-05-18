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
    * malloc the materials
    * and point the materials to their respective arays
    */
    size_t materialBuffSize = sumMatArr(matIn);
    std::unordered_map<uintptr_t, TextInfo *> textureTranslate;
    handleCudaError(cudaMalloc((void **)&matDev, materialBuffSize));

    //copy over the buffers
    MatGpu * matHost = new MatGpu[matIn.size()];

    for (size_t i = 0; i < matIn.size(); i++) {
        matHost[i].TextureArr[0] = matIn[i].getGpuTextures()[0];
        matHost[i].TextureArr[1] = matIn[i].getGpuTextures()[1];
        matHost[i].TextureArr[2] = matIn[i].getGpuTextures()[2];
        matHost[i].TextureArr[3] = matIn[i].getGpuTextures()[3];
        matHost[i].TextureArr[4] = matIn[i].getGpuTextures()[4];
    }
    
    handleCudaError(cudaMemcpy(matDev, matHost, matIn.size() * sizeof(MatGpu), cudaMemcpyHostToDevice));
    matLen = matIn.size();
    delete[] matHost;
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
        printf("Material idx : %d", static_cast<int>(cur.matIdx));
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
        printf("Printing Diffuse for material : %d \n", static_cast<int>(i));
        printf("Width : %d Height : %d Texture Address : %p \n",
        sceneInfo->matDev[i].TextureArr[0]->width, 
        sceneInfo->matDev[i].TextureArr[0]->height, 
        sceneInfo->matDev[i].TextureArr[0]->arr);
        printTextures(sceneInfo->matDev[i].TextureArr[0]);
    }
}