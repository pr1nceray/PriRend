#include "./Materials.h"



float Material::getLambertian(const Ray & ray, const glm::vec3 & normal) const {
    return std::fabs(glm::dot(ray.Dir, normal));
} 

glm::vec3 Material::generateNewDir(const Ray & rayIn, const glm::vec3 & normal) const {
    switch (materialOpt) {
        case (0) : {
            glm::vec3 randVec = generateRandomVecH();
            if(glm::dot(randVec, normal) < 0) {
                randVec *= -1.0f;
            }
            return randVec;
        }
        case (1) : {
            return (rayIn.Dir - (2.0f * (glm::dot(rayIn.Dir, normal)) * normal));
        }
        case (2) : {

        }
    }
    return glm::vec3(0,0,0);
}

const Color Material::bsdf(const Ray & rayIn, const Ray & rayOut, const glm::vec3 & normal) const {
    //glm::vec3 newRayDir = generateNewDir(rayIn, normal);

    float lambertian = getLambertian(rayIn, normal);

    return Color(1.0f, 1.0f, 1.0f);


}

const Color Material::brdf(const Ray & rayIn, const Ray & rayOut) const {
    return Color(1.0f, 1.0f, 1.0f);
}


Color Material::getDiffuse() const {
    return DiffuseBasic;
}

Color Material::getAmbient() const {
    return ambientBasic;
}

Color Material::getSpecular() const {
    return SpecularBasic;
}

void Material::setDiffuse(const Color & colorIn) {
    DiffuseBasic = colorIn;
}
void Material::setAmbient(const Color & colorIn) {
    ambientBasic = colorIn;
}
void Material::setSpecular(const Color & colorIn) {
    SpecularBasic = colorIn;
}