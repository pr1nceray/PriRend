#pragma once
#include "Object.h"
#include "Camera.h"
#include <vector>
//#include "Renderer.h>

class Scene
{
    private:
    std::vector<Object> Scene_Objects;
    Camera cam;
    
    public:

    void render() {
        cam.draw(Scene_Objects);
    }

    Object & add_object(std::string obj_name) {
        Scene_Objects.push_back(Object(obj_name));
        return Scene_Objects.back();
    }

    std::vector<Object> & getObjects() {
        return Scene_Objects;
    }
};