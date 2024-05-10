#include "scene.h"


int main() {
    Scene s = Scene();
    s.add_object("./assets/sphere.obj");
    s.add_object("./assets/cubestretch.obj");
    s.render();
}
