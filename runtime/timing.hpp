#include "pairs.hpp"

#pragma once

using namespace std;

namespace pairs {

void register_timer(PairsRuntime *ps, int id, std::string name) {
    ps->getTimers()->add(id, name);
}

void start_timer(PairsRuntime *ps, int id) {
    ps->getTimers()->start(id);
}

void stop_timer(PairsRuntime *ps, int id) {
    ps->getTimers()->stop(id);
}

void print_timers(PairsRuntime *ps) {
    ps->printTimers();
}

}
