#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assimp/scene.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include <glm/vec3.hpp>
#include "./Primitives.cuh"
#include "./Color.cuh"
#include "./Gpu/GpuPrimitives.cuh"

#define CHANNELSTEXTURE 4
/*
* TODO : figure out materials.
*/
class Material {
    public:
    explicit Material() {
        for(size_t i = 0; i < 5; ++i) {
            textures[i] = nullptr;
        }
        for(size_t i = 0; i < 5; ++i) {
            texturesDev[i] = nullptr;
        }
    }

    static const std::unordered_map<std::string, TextInfo *> & getTextures();
    static const std::vector<TextInfo *>getAllInfo();
    /*
    * Class functions
    */
    void loadTexture(const std::string & fileName, aiTextureType type);
    void setBasic(Color c, aiTextureType type) ;
    /*
    * Getters / Setters
    */

    const TextInfo * getDiffuse() const;
    const TextInfo * getNormal() const;
    const TextInfo * getSpecular() const;
    const TextInfo * getMetallic() const;
    const TextInfo * getRoughness() const;
    
    TextInfo ** getGpuTextures() const;
    TextInfo ** getHostTextures() const;
    
    private :
    TextInfo *loadImage(const std::string & fileName);
    void checkInScene(const std::string & fileName, size_t idx);

    /*
    * Order of materials : 
    * Diffuse, Normal, Specular, Metallic, Roughness
    * TexturesDev is textures, but on the gpu.
    */
    TextInfo * textures[5];
    TextInfo * texturesDev[5];
    static std::unordered_map<std::string, TextInfo *> currentMaterials;
    static std::vector<TextInfo *> allInfo;
    void convert(uint8_t * source, size_t max, float * out);
    void flipImage(uint8_t *imageData, size_t width, size_t height);
    void setColorToTextInfo(Color & c, size_t idx);
    void createCudaTexture(TextInfo * txtIn, float * dataIn, size_t width, size_t height);
};

