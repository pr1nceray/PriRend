#include "./Materials.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::unordered_map<std::string, TextInfo *> Material::currentMaterials;
std::unordered_map<uintptr_t, TextInfo *> Material::GpuMaterials;
std::vector<TextInfo *> TextInfoDelete;
std::vector<float *> TexturesDelete;

// quick way to map aiTextureType enum to size_t
const std::unordered_map<aiTextureType, size_t> textureEnumToIndex {
{aiTextureType_DIFFUSE, 0}, {aiTextureType_NORMALS, 1}, {aiTextureType_SPECULAR, 2}, 
{aiTextureType_METALNESS, 3},  {aiTextureType_DIFFUSE_ROUGHNESS, 4}};

const std::unordered_map<std::string, TextInfo *> & Material::getTextures() {
    return currentMaterials;
}

const std::vector<TextInfo *> & Material::getTextInfoDelete() {
    return TextInfoDelete;
}
const std::vector<float *> & Material::getTexturesDelete() {
    return TexturesDelete;
}

void Material::setBasic(Color c, aiTextureType type) {
    auto it = textureEnumToIndex.find(type);
    if(it == textureEnumToIndex.end()) {
        throw std::runtime_error("Attempting to load a basic texture of unknown type");
    }
    if(textures[it->second] != nullptr) {
        throw std::runtime_error("Attempting to set a texture to basic when it already has a value.");
    }
    textures[it->second] = new TextInfo();
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

TextInfo Material::loadImage(const std::string & fileName) {
    int width, height, numChannel;
    uint8_t * imageData = stbi_load(std::string("./assets/" + fileName).c_str(), & width, &height, &numChannel, 3);
    if (imageData == NULL || stbi_failure_reason()) {
        throw std::runtime_error("Error loading Texture file " + fileName  + ". See logs for more.");
    }
    float * newImageData = new float[width * height * 3];

    flipImage(imageData, width, height);
    convert(imageData, width * height * 3, newImageData);
    stbi_image_free(imageData);
    return TextInfo{newImageData, width, height};
}

void Material::checkInScene(const std::string & fileName, size_t idx) {
    if (currentMaterials.find(fileName) != currentMaterials.end()) {
        textures[idx] = currentMaterials.find(fileName)->second;
        texturesDev[idx] = GpuMaterials.find(reinterpret_cast<uintptr_t>(textures[idx]))->second;
        return;
    }

    // allow for host file image loading
    TextInfo *texture = new TextInfo();
    *texture = loadImage(fileName);
    currentMaterials[fileName] = texture;

    // clone the image, but on gpu
    float * array;
    size_t size = sizeof(float) * texture->width * texture->height * 3;
    handleCudaError(cudaMalloc((void **)&array, size));
    handleCudaError(cudaMemcpy((void *)array, texture->arr, size, cudaMemcpyHostToDevice));
    TexturesDelete.push_back(array);

    // create a clone of the text info, but on gpu.
    TextInfo gpuCopy = *texture;
    TextInfo *gpuPtr;
    gpuCopy.arr = array;
    handleCudaError(cudaMalloc((void **)&texturesDev[idx], sizeof(TextInfo)));
    handleCudaError(cudaMemcpy((void *)gpuPtr, &gpuCopy, sizeof(TextInfo), cudaMemcpyHostToDevice));

    // create link between host and gpu.
    GpuMaterials[reinterpret_cast<uintptr_t>(texture)] = gpuPtr;
    TextInfoDelete.push_back(texturesDev[idx]);
    return;
}

// note : make cuda function
void Material::convert(uint8_t * source, size_t max, float * out) {
    for(size_t i = 0; i < max; ++i) {
        out[i] = source[i]/255.0f;
    }
}

// note : make cuda function
void Material::flipImage(uint8_t *imageData, size_t width, size_t height) {
    for (size_t i = 0; i < height/2; ++i) {
        size_t idxNorm = 3 * (i * width);
        size_t idxSwap = 3 * (height - (i +1)) * width;
        for(size_t j = 0; j < width; ++j) {
            std::swap(imageData[idxNorm + (j * 3)], imageData[idxSwap + (j * 3)]);
            std::swap(imageData[idxNorm + (j * 3) + 1], imageData[idxSwap + (j * 3) +1]);
            std::swap(imageData[idxNorm + (j * 3) + 2], imageData[idxSwap + (j * 3) + 2]);
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

void Material::setColorToTextInfo(Color & c, size_t idx) {
    textures[idx]->basic = true;
    *(textures[idx]->basicColor) = c.r;
    *(textures[idx]->basicColor + 1) = c.g;
    *(textures[idx]->basicColor + 2) = c.b;
    handleCudaError(cudaMalloc((void **)texturesDev[idx], sizeof(TextInfo)));
    handleCudaError(cudaMemcpy((void *)texturesDev[idx], textures[idx], 
    sizeof(TextInfo), cudaMemcpyHostToDevice));
    TextInfoDelete.push_back(texturesDev[idx]);
}