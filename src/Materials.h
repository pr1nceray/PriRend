#pragma once
#include <math.h>
#include <glm/vec3.hpp>
#include "./Primitives.h"
#include "./Color.h"


/*
* TODO : figure out materials.
*/
class Material {
    public:
    Material() : 
    DiffuseBasic(Color(.5f, .5f, .5f)), materialOpt(0) {
    }

    float getLambertian(const Ray & ray, const glm::vec3 & normal) const;
    const Color bsdf(const Ray & rayIn, const Ray & rayOut, const glm::vec3 & normal) const;
    const Color brdf(const Ray & rayIn, const Ray & rayOut) const;


    /*
    * Getters / Setters
    */
    Color getDiffuse() const;
    Color getAmbient() const;
    Color getSpecular() const;

    void setDiffuse(const Color & colorIn);
    void setAmbient(const Color & colorIn);
    void setSpecular(const Color & colorIn);

    private :
    glm::vec3 generateNewDir(const Ray & rayIn, const glm::vec3 & normal) const;

    Color ambientBasic;
    Color DiffuseBasic;
    Color SpecularBasic;
    size_t materialOpt;

    float * Ambient;
    float * Diffuse;
    float * Specular;

};
