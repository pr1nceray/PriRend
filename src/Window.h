#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstring>

const size_t Width = 800;
const size_t Height = 600;

const std::vector<const char *> validationLayers =
{
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


bool check_validation_layer_support();

class Window
{
    public:
    
    Window()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(Width, Height, "Vulkan", nullptr, nullptr);
        initVulkan();

    }

    ~Window()
    {
        clean_up();
    }

    private:
    GLFWwindow * window;
    VkInstance instance;

    void mainLoop()
    {
        while(!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
        }
    }

    void clean_up()
    {
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void initVulkan()
    {
        create_vulkan_instance();
        pickPhysicalDevice();
    }

    void create_vulkan_instance()
    {
        if(enableValidationLayers && !check_validation_layer_support())
        {
            throw std::runtime_error("Validation layer requested, but not available.");
        }

        VkApplicationInfo appInfo{};
        appInfo.pApplicationName = "Pri-Render";
        appInfo.applicationVersion = VK_MAKE_VERSION(0,0,1);
        appInfo.pEngineName = "Pri-Render";
        appInfo.engineVersion = VK_MAKE_VERSION(0,0,1);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtenCount = 0;
        const char ** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtenCount);
        createInfo.enabledExtensionCount = glfwExtenCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;
        
        if(enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else{
            createInfo.enabledLayerCount = 0;
        }
        
        if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create instance.");
        }
    }
    void pickPhysicalDevice()
    {
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count,nullptr);
        if(!dev_count)
        {
            throw std::runtime_error("No compatible devices.");
        }
        std::vector<VkPhysicalDevice> devices(dev_count);
        vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

        for(const auto & device : devices)
        {
            if(isDeviceSuitable(device))
            {   
                physicalDevice = device;
                break;
            }
        }

        if(physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable device.");
        }
    }


};

bool check_validation_layer_support()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount,nullptr);

    std::vector<VkLayerProperties> available(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, available.data());
    
    for(const char * layerName : validationLayers)
    {
        bool layer_found = false;
        for(const auto & layerProp : available)
        {
            if(strcmp(layerName, layerProp.layerName) == 0)
            {
                layer_found = true;
                break;
            }
        }
        if(!layer_found)
        {
            return false;
        }
    }
    return true;
}


bool isDeviceSuitable(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    return ((deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) && deviceFeatures.geometryShader);
}

int rateDeviceSuitability(VkPhysicalDevice device)
{
    
}