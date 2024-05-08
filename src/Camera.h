#pragma once
#include "object.h"
#include "ray.h"
#include "stb_image_write.h"
#include <vector>

#define WIDTH 300
#define HEIGHT 200

#define FOV_Y 60
#define FOV_X 90

class Camera
{

    public:
    
    Camera()
    {
        Final_image.resize(WIDTH * HEIGHT);
    }

    void draw(const std::vector<object> & obs)
    {
        for(size_t i = 0; i < HEIGHT; ++i)
        {
            for(size_t j = 0; j < WIDTH; ++j)
            {
                //spawn_ray(i * WIDTH + j, FOV_Y, FOV_X);
                Final_image[i * WIDTH + j] = glm::vec3(static_cast<float>(i)/WIDTH, static_cast<float>(j)/HEIGHT, 1.0f);
            }
        }
        stbi_write_png("Output.png", WIDTH, HEIGHT, 3, Final_image.data(), WIDTH * 3);
    }

    private:
    glm::vec4 cent;
    glm::vec4 rot;
    std::vector<glm::vec3> Final_image;
};