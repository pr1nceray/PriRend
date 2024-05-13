#include "scene.cuh"
#include <cstdlib>

int main() {
    srand (static_cast <unsigned> (time(0)));
    Scene s = Scene();
    //s.add_object("./assets/sphere1.obj");
    //s.add_object("./assets/sphere2.obj");
    //s.add_object("./assets/plane1.obj");
    s.add_object("./assets/suzanne.obj");
    s.render();
}
