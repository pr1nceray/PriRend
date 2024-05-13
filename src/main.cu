#include "scene.cuh"
#include <cstdlib>

int main() {
    srand (static_cast <unsigned> (time(0)));
    Scene s = Scene();
    s.add_object("./assets/cubePlane.obj");
    s.render();
}
