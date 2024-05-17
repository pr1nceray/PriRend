#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "./Object.cuh"
#include "./Camera.cuh"
#include "./Materials.cuh"
#include "./Gpu/GpuInfo.cuh"

class Scene
{
    private:
    std::vector<Object> Scene_Objects;
    std::vector<Mesh> sceneMeshs;
    std::vector<Material> sceneMats;

    Camera cam;
    
    /*
    * Object info not needed, since it is mostly used
    * For translation, rotation, etc and i havent gotten there yet.
    */
    GpuInfo prepareMesh() {
        for (size_t i = 0; i < Scene_Objects.size(); ++i) {
            const Object & curObj = Scene_Objects[i];
            for (size_t j = 0; j < curObj.getObjInfo().size(); j++) {
                sceneMeshs.push_back(curObj.getObjInfo()[j]);
            }
        }
        GpuInfo temp = GpuInfo(sceneMeshs, sceneMats);
        // printMeshGlobal<<<1,1>>>();
        // printMaterialInfo<<<1,1>>>();
        return temp;
    }
    
    void freeAllMaterials() {
        // deletes the basic textures
        for (auto it : sceneMats) {
            freeBasic(it.getDiffuse());
            freeBasic(it.getNormal());
            freeBasic(it.getSpecular());
            freeBasic(it.getMetallic());
            freeBasic(it.getRoughness());
        }

        // delete the image textures that we allocated
        for (auto it : Material::getTextures()) {
            delete[] it.second->arr; // free information.
            delete it.second; // delete textureinfo ptr
        }
    }

    void freeBasic(const TextInfo * ptr) {
        if(ptr->basic) {
            delete ptr;
        }
    } 

    public:
    explicit Scene() = default;
    ~Scene() {
        freeAllMaterials();
    }

    void render() {
        GpuInfo gpu = prepareMesh();
        cam.drawProgressive();
        gpu.freeResources();
    }

    Object & add_object(std::string obj_name) {
        Scene_Objects.push_back(Object(obj_name, sceneMats));
        return Scene_Objects.back();
    }
    
    std::vector<Object> & getObjects() {
        return Scene_Objects;
    }
};
