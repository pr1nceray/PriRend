#include "Primitives.h"
#include <iostream>


int main() {
    //Scene s = Scene();
    //s.add_object("./assets/sphere.obj");
    //s.add_object("./assets/cubestretch.obj");
    //s.render();
    for (size_t i = 0; i < 5; ++i) {
        std::cout << static_cast<int>(generateRandomNum()) << "\n";
    }
}
