#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

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
    Diffuse(nullptr), Normal(nullptr), Specular(nullptr) {

    }

    static const std::unordered_map<std::string, TextInfo *> & getTextures();
    TextInfo loadImage(const std::string & fileName);
    void loadDiffuse(const std::string & fileName);
    void loadNormal(const std::string & fileName);
    void loadSpecular(const std::string & fileName);
    
    /*
    * Class functions
    */
    float getLambertian(const Ray & ray, const glm::vec3 & normal) const;

    void freeResources(); 

    /*
    * Getters / Setters
    */

    const TextInfo * getDiffuse() const;
    private :
    TextInfo *checkInScene(const std::string & fileName);

    TextInfo *Diffuse;
    TextInfo *Normal;
    TextInfo *Specular;

    static std::unordered_map<std::string, TextInfo *> currentMaterials;
    void convert(uint8_t * source, size_t max, float * out);
    void flipImage(uint8_t *imageData, size_t width, size_t height);
};

