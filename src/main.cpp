#include "scene.h"
#include <cstdlib>

int main() {
    srand (static_cast <unsigned> (time(0)));
    Scene s = Scene();
    s.add_object("./assets/suzanne.obj");

    //s.getObjects()[0].setMeshColors(Color())
    s.render();
}
