#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assimp/scene.h>

#include <string>
#include <stdexcept>
#include <unordered_map>

#include <glm/vec3.hpp>
#include "./Primitives.cuh"
#include "./Color.cuh"


/*
* TODO : figure out materials.
*/
class Material {
    public:
    Material() : 
    Diffuse(nullptr), Normal(nullptr), Specular(nullptr), 
    Metallic(nullptr), Roughness(nullptr),
    diffBasic(false), NormalBasic(false), SpecularBasic(false), 
    MetallicBasic(false), RoughnessBasic(false) {
    }

    static const std::unordered_map<std::string, TextInfo *> & getTextures();
    /*
    * Class functions
    */
    void loadTexture(const std::string & fileName, aiTextureType type);
    void setBasic(Color c, aiTextureType type) ;
    /*
    * Getters / Setters
    */

    const TextInfo * getDiffuse() const;
    private :
    TextInfo loadImage(const std::string & fileName);
    TextInfo *checkInScene(const std::string & fileName);

    /*
    * The boolean and Color values here could be stored 
    * Inside the textInfo object. However, this would mean
    * Allocating memory for the textuinfo object ahead of time(constructor)
    * which would defeat the purpouse of sharing TextInfo ptr
    * You could theoretically allocate memory on launch
    * then delete it when overwriting it, but that just seems
    * like its too much work considering the added memory complexity
    * 
    */
    TextInfo *Diffuse;
    bool diffBasic;
    Color diffBasicColor;

    TextInfo *Normal;
    bool NormalBasic;
    Color normalBasicColor;

    TextInfo *Specular;
    bool SpecularBasic;
    Color SpecularBasicColor;

    TextInfo *Metallic;
    bool MetallicBasic;
    Color MetallicBasicColor;

    TextInfo *Roughness;
    bool RoughnessBasic;
    Color RoughnessBasicColor;

    static std::unordered_map<std::string, TextInfo *> currentMaterials;
    void convert(uint8_t * source, size_t max, float * out);
    void flipImage(uint8_t *imageData, size_t width, size_t height);
};

