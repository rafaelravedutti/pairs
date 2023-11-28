#include "pairs.hpp"

#pragma once

using namespace std;

namespace pairs {

void register_timer(PairsSimulation *ps, int id, std::string name) {
    ps->getTimers()->add(id, name);
}

void start_timer(PairsSimulation *ps, int id) {
    ps->getTimers()->start(id);
}

void stop_timer(PairsSimulation *ps, int id) {
    ps->getTimers()->stop(id);
}

void print_timers(PairsSimulation *ps) {
    ps->printTimers();
}

}
