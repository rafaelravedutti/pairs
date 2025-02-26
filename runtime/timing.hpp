#include "pairs.hpp"

#pragma once

using namespace std;

namespace pairs {

void register_timer(PairsRuntime *ps, int id, std::string name);
void start_timer(PairsRuntime *ps, int id);
void stop_timer(PairsRuntime *ps, int id);
void print_timers(PairsRuntime *ps);

}
