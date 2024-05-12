#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "./Object.cuh"
#include "./Camera.cuh"
#include "./GpuInfo.cuh"

class Scene
{
    private:
    std::vector<Object> Scene_Objects;
    std::vector<Mesh> sceneMeshs;
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
        GpuInfo temp = GpuInfo(sceneMeshs);
        printMeshInfo<<<1,1,1>>>(temp);
        return temp;
    }
    
    public:

    void render() {
        GpuInfo gpu = prepareMesh();
        cam.draw(gpu);
        gpu.freeResources();
    }

    Object & add_object(std::string obj_name) {
        Scene_Objects.push_back(Object(obj_name));
        return Scene_Objects.back();
    }

    std::vector<Object> & getObjects() {
        return Scene_Objects;
    }
};
