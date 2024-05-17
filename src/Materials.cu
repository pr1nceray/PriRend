#include "./Materials.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>

std::unordered_map<std::string, TextInfo *> Material::currentMaterials;

const std::unordered_map<std::string, TextInfo *> & Material::getTextures() {
    return currentMaterials;
}

void Material::setBasic(Color c, aiTextureType type) {
    switch (type) {
        case (aiTextureType_DIFFUSE) : {
            diffBasicColor = c;
            diffBasic = true;
            break;
        }
        case(aiTextureType_NORMALS) : {
            normalBasicColor = c;
            NormalBasic = true;
            break;
        }
        case(aiTextureType_SPECULAR) : {
            SpecularBasicColor = c;
            SpecularBasic = true;
            break;
        }
        case(aiTextureType_METALNESS) : {
            MetallicBasicColor = c;
            MetallicBasic = true;
            break;
        }
        case(aiTextureType_DIFFUSE_ROUGHNESS) : {
            RoughnessBasicColor = c;
            RoughnessBasic = true;
            break;
        }
        default : { 
            throw std::runtime_error("Found unknown basic texture type");
        }
    }
}
void Material::loadTexture(const std::string & fileName, aiTextureType type) {
    switch (type) {
        case (aiTextureType_DIFFUSE) : {
            Diffuse = checkInScene(fileName);
            break;
        }
        case(aiTextureType_NORMALS) : {
            Normal = checkInScene(fileName);
            break;
        }
        case(aiTextureType_SPECULAR) : {
            Specular = checkInScene(fileName);
            break;
        }
        case(aiTextureType_METALNESS) : {
            Metallic = checkInScene(fileName);
            break;
        }
        case(aiTextureType_DIFFUSE_ROUGHNESS) : {
            Roughness = checkInScene(fileName);
            break;
        }
        default : { 
            throw std::runtime_error("Found unknown texture type");
        }
    }
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

TextInfo *Material::checkInScene(const std::string & fileName) {
    if (Material::currentMaterials.find(fileName) != Material::currentMaterials.end()) {
        return Material::currentMaterials.find(fileName)->second;
    }
    TextInfo *texture = new TextInfo();
    *texture = loadImage(fileName);
    Material::currentMaterials[fileName] = texture;
    return texture;
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
    return Diffuse;
}

