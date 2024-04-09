#include <iostream>
//---
#include "md.hpp"

int main(int argc, char **argv) {
    PairsSimulation *ps = new PairsSimulation();
    std::cout << "initialize" << std::endl;
    ps->initialize(argc, argv);
    std::cout << "do_timestep" << std::endl;
    ps->do_timestep();
    std::cout << "end" << std::endl;
    ps->end();
    return 0;
}
