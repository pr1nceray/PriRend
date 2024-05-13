#include "./GpuInfo.cuh"

__device__ Ray MeshGpu::generateRandomVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const {
    glm::vec3 randVec = generateRandomVecD(state);
    glm::vec3 normal = getFaceNormal(faceIdx);
    randVec *= glm::dot(randVec, normal) < 0?-1:1;

    glm::vec3 newOrigin = origin + (randVec * .001f); //avoid shadow acne
    return Ray(newOrigin, randVec);
}

__device__ Ray MeshGpu::generateLambertianVecOnFace(const size_t faceIdx, curandState * state, const glm::vec3 & origin) const {
    glm::vec3 newDir = getFaceNormal(faceIdx) + generateRandomVecD(state);
    glm::vec3 newOrigin = origin + (newDir * .01f); // avoid shadow acne
    return Ray(newOrigin, newDir);
}

__device__ Ray MeshGpu::generateReflectiveVecOnFace(const size_t faceIdx, const glm::vec3 & dir, const glm::vec3 & origin) const {
    const glm::vec3 & normal = getFaceNormal(faceIdx);
    const glm::vec3 newDir = dir - (2 * glm::dot(normal, dir) * normal);
    const glm::vec3 newOrigin = origin + (newDir * .01f); // avoid shadow acne
    return Ray(newOrigin, newDir);
}


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
        // face normals
        total += sizeof(glm::vec3) * meshIn[i].FaceNormals.size();
        // edgemap
        total += sizeof(glm::vec3) * meshIn[i].EdgeMap.size();
        //faces
        total += sizeof(glm::ivec3) * meshIn[i].Faces.size();
        // V buffer.
        total += sizeof(Vertex) * meshIn[i].Indicies.size(); 
    }
    return total;
}

void GpuInfo::copyIntoDevice(const std::vector<Mesh> & meshIn) {
    meshLen = meshIn.size();

    size_t sizeOfMeshes = sumMeshSizes(meshIn); //size of the information
    cudaError_t err = cudaMalloc((void **)(&infoBuffer), sizeOfMeshes); //malloc the info
    handleCudaError(err);


    size_t sizeOfArray = sumMeshArr(meshIn); // size of the struct to hold the information

    err = cudaMalloc((void **)(&meshDev), sizeOfArray); //malloc the array of ptrs
    handleCudaError(err);

    MeshGpu * meshHost = new MeshGpu[sizeOfArray]; //create information holder on host

    /*
    * Copy over all the vertex buffers one after each other for cache purpouses.
    * Buffer copy is so that we dont lose infoBuffer.
    * take the size of the normalVector array, and copy over the data into where
    * BufferCopy is pointing. From there, set the pointer of mesh's to be accurate.
    * Then add the size that we copied.
    * same for TQ and edgemap.
    * 
    */
    void * bufferCpy = infoBuffer;
    copyNormalBuff(bufferCpy, meshIn, meshHost);
    copyEdgeBuff(bufferCpy, meshIn, meshHost);
    copyFaceBuff(bufferCpy, meshIn, meshHost);
    copyVertexBuff(bufferCpy, meshIn, meshHost);
    setLength(meshIn, meshHost);

    //copy over all the info
    err = cudaMemcpy(meshDev, meshHost, sizeOfArray, cudaMemcpyHostToDevice);
    handleCudaError(err);
    delete[] meshHost; //no longer needed, free resources.
}

 __host__ void GpuInfo::freeResources() {
    cudaFree(infoBuffer);
    cudaFree(meshDev);
 }

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

void GpuInfo::setLength( const std::vector<Mesh> & meshIn, MeshGpu * meshHost) {
    for(size_t i = 0; i < meshIn.size(); ++i) {
        meshHost[i].faceSize = meshIn[i].Faces.size();
        meshHost[i].vertexSize = meshIn[i].Indicies.size();
    }
}

size_t GpuInfo::sumMeshArr(const std::vector<Mesh> & meshIn) const {
    return sizeof(MeshGpu) * meshIn.size();
}


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