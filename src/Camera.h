#pragma once
#include <vector>
#include "./Object.h"
#include "./rayOps.h"
#include "./stb_image_write.h"

/*
* Camera is an object that has the ability to render 
* A scene. Currently, it only renders things above it on the z axis.
*/

class Camera {
    public:
    Camera() {
        Final_image.resize(WIDTH * HEIGHT * 3);
    }

    /*
    * Takes the objects in as arguments
    * Renders a scene given the objects in the Scene.
    */
    void draw(const std::vector<Object> & obs) {
        for (size_t y = 0; y < static_cast<size_t>(HEIGHT); ++y) {
            for (size_t x = 0; x < static_cast<size_t>(WIDTH); ++x) {
                Color final = spawnRay(x, y, FOV_Y, FOV_X , obs);
                size_t index = 3 * (y * WIDTH + x);

                final *= 255.0f;
                uint8_t finalR = final.r > 255? 255 : static_cast<uint8_t>(final.r);
                uint8_t finalG = final.g > 255? 255 : static_cast<uint8_t>(final.g);
                uint8_t finalB = final.b > 255? 255 : static_cast<uint8_t>(final.b);
                
                
                Final_image[index] = finalR;
                Final_image[index + 1] = finalG;
                Final_image[index + 2] = finalB;
            }
        }
        // Write image after done processing scene.
        Write_Image();
    }

    private:

    void Write_Image() {
        stbi_write_png("Output.png", WIDTH,
        HEIGHT, 3, Final_image.data(), WIDTH * 3);
    }

    glm::vec4 cent;
    glm::vec4 rot;
    std::vector<uint8_t> Final_image;
};
