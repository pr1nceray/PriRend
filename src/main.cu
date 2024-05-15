#include "scene.cuh"
#include <cstdlib>
#include <stdexcept>

int main() {
    srand (static_cast <unsigned> (time(0)));
    try {
        Scene s = Scene();
        s.add_object("./assets/noTextures.fbx");
        s.render();
    } catch(std::runtime_error err) {
        std::cerr << err.what();
    }

}
