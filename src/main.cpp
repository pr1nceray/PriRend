#include "scene.h"


int main()
{
    Scene s = Scene();
    s.add_object("./assets/sphere.obj");
    s.render();
}

/*
int main()
{
    Renderer app;
    try
    {
        app.run();
    }
    catch (const std::exception & err)
    {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
*/