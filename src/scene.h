#pragma once
#include "object.h"
#include "Camera.h"
#include <vector>
//#include "Renderer.h>

class Scene
{
    private:
    std::vector<object> Scene_Objects;
    Camera cam;
    
    public:

    void render()
    {
        cam.draw(Scene_Objects);
    }

    void add_object(std::string obj_name)
    {
        Scene_Objects.push_back(object(obj_name));
    }
};