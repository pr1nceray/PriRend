#pragma once
#include "object.h"
#include "ray.h"
#include "stb_image_write.h"

#include <vector>



class Camera
{

    public:
    
    Camera()
    {
        Final_image.resize(WIDTH * HEIGHT * 3);
    }

    void draw(const std::vector<object> & obs)
    {
        for(size_t y = 0; y < static_cast<size_t>(HEIGHT); ++y)
        {
            for(size_t x = 0; x < static_cast<size_t>(WIDTH); ++x)
            {
                Color final = spawnRay(x, y, FOV_Y, FOV_X , obs);
                
                size_t index = 3 * (y * WIDTH + x);
                //std::cout << index << "\n";
                Final_image[index] = 250.0f * static_cast<float>(y)/HEIGHT; 
                Final_image[index + 1]  = 250.0f * static_cast<float>(x)/WIDTH;
                Final_image[index + 2]  = 250.0f;
            }
        }
        stbi_write_png("Output.png", WIDTH, HEIGHT, 3, Final_image.data(), WIDTH * 3);
    }

    private:
    glm::vec4 cent;
    glm::vec4 rot;
    std::vector<uint8_t> Final_image;
};