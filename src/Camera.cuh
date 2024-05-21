#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "./Object.cuh"
#include "./Gpu/rayOps.cuh"
#include "./stb_image_write.h"
#include "./Gpu/GpuInfo.cuh"

/*
* Camera is an object that has the ability to render 
* A scene. Currently, it only renders things above it on the z axis.
*/

class Camera {
    public:
    Camera() : 
    cent(0, 0, 0, 0), rot(0, 0 ,0 ,0), imageHost(nullptr), imageDev(nullptr) {
        // used for host writing images
        const size_t sizePixel = CHANNEL * WIDTH * HEIGHT;
        sizeImage = sizePixel * sizeof(uint8_t);
        imageHost = new uint8_t[sizePixel];
        handleCudaError(cudaMalloc((void **)&imageDev, sizePixel * sizeof(uint8_t)));
        handleCudaError(cudaMalloc((void **)&progressiveArr, sizePixel * sizeof(float)));
        block = dim3(32,32,1);
    }

    ~Camera() {
        delete[] imageHost;
        cudaFree(imageDev);
        cudaFree(progressiveArr);
    }
    /*
    * Takes the objects in as arguments
    * Renders a scene given the objects in the Scene.
    */
    void draw() {
                
        size_t gridx = (WIDTH/32) + (WIDTH%32>0?1:0);
        size_t gridy = (HEIGHT/32) + (HEIGHT%32>0?1:0);

        dim3 grid = dim3(gridx, gridy, 1);  
        
        int seed = rand();
    
        spawnRay<<<grid, block>>>(seed, static_cast<uint8_t *>(imageDev));
        handleCudaError(cudaGetLastError());
        handleCudaError(cudaDeviceSynchronize());
        handleCudaError(cudaMemcpy(imageHost, imageDev, sizeImage, cudaMemcpyDeviceToHost));

        Write_Image();
    }

        void drawProgressive() {
        
        size_t gridx = (WIDTH/32) + (WIDTH%32>0?1:0);
        size_t gridy = (HEIGHT/32) + (HEIGHT%32>0?1:0);

        dim3 grid = dim3(gridx, gridy, 1);  

        // Clear array since adding
        wipeArr<<<grid, block>>>(progressiveArr);
        for (size_t i = 0; i < SPP; ++i) {
            
            // generate new seed, since its being passed as a param 
            int seed = rand();
    
            spawnRayProgressive<<<grid, block>>>(seed, progressiveArr);
            handleCudaError(cudaGetLastError());
            handleCudaError(cudaDeviceSynchronize());

            if (i % 16 == 0) {
                convertArr<<<grid, block>>>(progressiveArr, imageDev);
                handleCudaError(cudaGetLastError());
                handleCudaError(cudaDeviceSynchronize());
                handleCudaError(cudaMemcpy(imageHost, imageDev, sizeImage, cudaMemcpyDeviceToHost));
                Write_Image();
            }
        }
        handleCudaError(cudaDeviceSynchronize());
        convertArr<<<grid, block>>>(progressiveArr, imageDev);
        handleCudaError(cudaGetLastError());
        handleCudaError(cudaDeviceSynchronize());
        handleCudaError(cudaMemcpy(imageHost, imageDev, sizeImage, cudaMemcpyDeviceToHost));
        Write_Image();
    }

    private:
    glm::vec4 cent;
    glm::vec4 rot;

    uint8_t * imageHost;
    uint8_t * imageDev; 
    float * progressiveArr;

    size_t sizeImage;
    dim3 block;
    void Write_Image() {
        stbi_write_png("Output.png", WIDTH, HEIGHT, CHANNEL, imageHost, WIDTH * CHANNEL);
    }


};
