#include "./Materials.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::unordered_map<std::string, TextInfo *> Material::currentMaterials;
std::vector<TextInfo *> Material::allInfo;

// quick way to map aiTextureType enum to size_t
const std::unordered_map<aiTextureType, size_t> textureEnumToIndex {
{aiTextureType_DIFFUSE, 0}, {aiTextureType_NORMALS, 1}, {aiTextureType_SPECULAR, 2}, 
{aiTextureType_METALNESS, 3},  {aiTextureType_DIFFUSE_ROUGHNESS, 4}, {aiTextureType_EMISSION_COLOR, 5},
{aiTextureType_EMISSIVE, 6}};

const std::unordered_map<std::string, TextInfo *> & Material::getTextures() {
    return currentMaterials;
}

const std::vector<TextInfo *>Material::getAllInfo() {
    return allInfo;
}

void Material::setBasic(Color c, aiTextureType type) {
    auto it = textureEnumToIndex.find(type);
    if(it == textureEnumToIndex.end()) {
        throw std::runtime_error("Attempting to load a basic texture of unknown type");
    }
    if(textures[it->second] != nullptr) {
        throw std::runtime_error("Attempting to set a texture to basic when it already has a value.");
    }
    setColorToTextInfo(c, it->second);
}

void Material::loadTexture(const std::string & fileName, aiTextureType type) {
    auto it = textureEnumToIndex.find(type);
    if(it == textureEnumToIndex.end()) {
        throw std::runtime_error("Attempting to load an image texture of unknown type");
    }
    if(textures[it->second] != nullptr) {
        throw std::runtime_error("Attempting to set a texture to Loaded image when it already has a value.");
    }
    checkInScene(fileName, it->second);
}

TextInfo* Material::loadImage(const std::string & fileName) {
    int width, height, numChannel;
    uint8_t * imageData = stbi_load(std::string("./assets/" + fileName).c_str(), 
    & width, &height, &numChannel, CHANNELSTEXTURE);

    if (imageData == NULL) {
        throw std::runtime_error("Error loading Texture file " + fileName + "." +
        std::string(stbi_failure_reason()));
    }
    float * newImageData = new float[CHANNELSTEXTURE * width * height];
    flipImage(imageData, width, height);
    convert(imageData, width * height * CHANNELSTEXTURE, newImageData);
    stbi_image_free(imageData);
    TextInfo *texture = new TextInfo();
    texture->basic = false;
    createCudaTexture(texture, newImageData, width, height);
    delete[] newImageData;
    return texture;
}

void Material::checkInScene(const std::string & fileName, size_t idx) {
    if (currentMaterials.find(fileName) != currentMaterials.end()) {
        textures[idx] = currentMaterials.find(fileName)->second;
        return;
    }

    TextInfo *texture = loadImage(fileName);
    currentMaterials[fileName] = texture;
    textures[idx] = texture;
    
    handleCudaError(cudaMalloc((void **)&texturesDev[idx], sizeof(TextInfo)));
    handleCudaError(cudaMemcpy((void *)texturesDev[idx], textures[idx], sizeof(TextInfo), cudaMemcpyHostToDevice));
    allInfo.push_back(texture);
    return;
}

//note : flip and convert function?
// note : make cuda function
void Material::convert(uint8_t * source, size_t max, float * out) {
    for(size_t i = 0; i < max; ++i) {
        out[i] = source[i]/255.0f;
    }
}

// note : make cuda function
void Material::flipImage(uint8_t *imageData, size_t width, size_t height) {
    for (size_t i = 0; i < height/2; ++i) {
        size_t idxNorm = CHANNELSTEXTURE * (i * width);
        size_t idxSwap = CHANNELSTEXTURE * (height - (i +1)) * width;
        for(size_t j = 0; j < width; ++j) {
            std::swap(imageData[idxNorm + (j * CHANNELSTEXTURE)],
             imageData[idxSwap + (j * CHANNELSTEXTURE)]);
            std::swap(imageData[idxNorm + (j * CHANNELSTEXTURE) + 1],
             imageData[idxSwap + (j * CHANNELSTEXTURE) +1]);
            std::swap(imageData[idxNorm + (j * CHANNELSTEXTURE) + 2], 
            imageData[idxSwap + (j * CHANNELSTEXTURE) + 2]);
            std::swap(imageData[idxNorm + (j * CHANNELSTEXTURE) + 3], 
            imageData[idxSwap + (j * CHANNELSTEXTURE) + 3]);
        }
    }
}

const TextInfo * Material::getDiffuse() const {
    return textures[0];
}

const TextInfo * Material::getNormal() const {
    return textures[1];
}

const TextInfo * Material::getSpecular() const {
    return textures[2];
}

const TextInfo * Material::getMetallic() const {
    return textures[3];
}

const TextInfo * Material::getRoughness() const {
    return textures[4];
}
TextInfo ** Material::getGpuTextures() const {
    return (TextInfo**)texturesDev;
}

TextInfo ** Material::getHostTextures() const {
    return (TextInfo**)textures;
}

void Material::setColorToTextInfo(Color & c, size_t idx) {
    textures[idx] = new TextInfo();
    textures[idx]->basic = true;
    (textures[idx]->basicColor.x) = c.r;
    (textures[idx]->basicColor.y) = c.g;
    (textures[idx]->basicColor.z) = c.b;
    handleCudaError(cudaMalloc((void **)&texturesDev[idx], sizeof(TextInfo)));
    handleCudaError(cudaMemcpy((void *)texturesDev[idx], textures[idx], 
    sizeof(TextInfo), cudaMemcpyHostToDevice));
    
    allInfo.push_back(textures[idx]);

}

void Material::createCudaTexture(TextInfo *txtIn, float * dataIn, size_t width, size_t height) {
    cudaArray_t dataGpu;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); 

    handleCudaError(cudaGetLastError());
    handleCudaError(cudaMallocArray(&dataGpu, &channelDesc, width, height));

    const size_t size = height * width * CHANNELSTEXTURE * sizeof(float);
    handleCudaError(cudaMemcpyToArray(dataGpu, 0, 0, dataIn, size, cudaMemcpyHostToDevice));
    
   cudaResourceDesc resDec;
   memset(&resDec, 0, sizeof(cudaResourceDesc));
   resDec.resType = cudaResourceTypeArray;
   resDec.res.array.array = dataGpu;

   cudaTextureDesc textDesc;
   memset(&textDesc, 0, sizeof(cudaTextureDesc));
   textDesc.addressMode[0] = cudaAddressModeBorder;
   textDesc.addressMode[1] = cudaAddressModeBorder;
   textDesc.filterMode = cudaFilterModeLinear;
   textDesc.readMode = cudaReadModeElementType;
   textDesc.normalizedCoords = 1;

   txtIn->text = 0;
   handleCudaError(cudaCreateTextureObject(&txtIn->text, &resDec, &textDesc, NULL));

}