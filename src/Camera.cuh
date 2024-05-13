#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "./Object.cuh"
#include "./rayOps.cuh"
#include "./stb_image_write.h"
#include "./GpuInfo.cuh"

/*
* Camera is an object that has the ability to render 
* A scene. Currently, it only renders things above it on the z axis.
*/

class Camera {
    public:
    Camera() : 
    cent(0, 0, 0, 0), rot(0, 0 ,0 ,0), imageHost(nullptr), imageDev(nullptr) {
        sizeImage = sizeof(uint8_t) * CHANNEL * WIDTH * HEIGHT;
        imageHost = new uint8_t[CHANNEL * WIDTH * HEIGHT];
        handleCudaError(cudaMalloc((void **)&imageDev, sizeImage));
    }

    ~Camera() {
        delete[] imageHost;
        cudaFree(imageDev);
    }
    /*
    * Takes the objects in as arguments
    * Renders a scene given the objects in the Scene.
    */
    void draw(GpuInfo info) {

        dim3 block = dim3(32,32,1);
        

        
        size_t gridx = (WIDTH/32) + (WIDTH%32>0?1:0);
        size_t gridy = (HEIGHT/32) + (HEIGHT%32>0?1:0);

        dim3 grid = dim3(gridx, gridy, 1);  
        
        int seed = rand();
        spawnRay<<<grid, block>>>(info, seed, imageDev);

        handleCudaError(cudaDeviceSynchronize());
        handleCudaError(cudaMemcpy(imageHost, imageDev, sizeImage, cudaMemcpyDeviceToHost));
        Write_Image();
    }

    private:
    glm::vec4 cent;
    glm::vec4 rot;
    uint8_t * imageHost;
    uint8_t * imageDev; 
    size_t sizeImage;

    void Write_Image() {
        stbi_write_png("Output.png", WIDTH, HEIGHT, 3, imageHost, WIDTH * 3);
    }

};
