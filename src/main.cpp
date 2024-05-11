#include "scene.h"
#include <cstdlib>

int main() {
    srand (static_cast <unsigned> (time(0)));
    Scene s = Scene();
    s.add_object("./assets/twoSphere.obj");

    s.add_object("./assets/sphere2.obj");
    //s.getObjects()[0].setMeshColors(Color())
    s.render();
}
