#pragma once
#include "Object.h"
#include "Camera.h"
#include <vector>
//#include "Renderer.h>

class Scene
{
    private:
    std::vector<Object> Scene_Objects;
    std::vector<Mesh> sceneMeshs;
    Camera cam;
    
    void prepareMesh() {
        for (size_t i = 0; i < Scene_Objects.size(); ++i) {
            const Object & curObj = Scene_Objects[i];
            for (size_t j = 0; j < curObj.getObjInfo().size(); j++) {
                sceneMeshs.push_back(curObj.getObjInfo()[j]);
            }
        }
    }
    
    public:

    void render() {
        prepareMesh();
        cam.draw(sceneMeshs);
    }

    Object & add_object(std::string obj_name) {
        Scene_Objects.push_back(Object(obj_name));
        return Scene_Objects.back();
    }

    std::vector<Object> & getObjects() {
        return Scene_Objects;
    }
};