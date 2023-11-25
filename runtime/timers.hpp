#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace pairs {

template<typename TimeType, typename TimeUnit = std::chrono::nanoseconds>
class Timers {
public:
    Timers(TimeType _factor) : time_factor(_factor) {}
    ~Timers() {}

    void add(size_t id, std::string name) {
        counter_names.resize(id + 1);
        counters.resize(id + 1);
        clocks.resize(id + 1);
        counter_names[id] = name;
    }

    void start(size_t id) { clocks[id] = std::chrono::high_resolution_clock::now(); }

    void stop(size_t id) {
        auto current_clock = std::chrono::high_resolution_clock::now();
        counters[id] += static_cast<TimeType>(
            std::chrono::duration_cast<TimeUnit>(current_clock - clocks[id]).count()) * time_factor;
    }

    void print() {
        std::unordered_map<std::string, TimeType> categorySums;

        std::cout << "all: " << counters[0] << std::endl;
        for (size_t i = 1; i < counters.size(); ++i) {
            const std::string& counterName = counter_names[i];
            TimeType counterValue = counters[i];

            if(counterName.find("pack_") == 0 ||
               counterName.find("unpack_") == 0 ||
               counterName.find("determine_") == 0 ||
               counterName.find("set_communication_") == 0 ||
               counterName.find("remove_exchanged_particles") == 0 ||
               counterName.find("change_size_after_exchange") == 0) {

                categorySums["communication"] += counterValue;

            } else if(counterName.find("build_cell_lists") == 0 ||
                      counterName.find("build_cell_lists_stencil") == 0 ||
                      counterName.find("partition_cell_lists") == 0 ||
                      counterName.find("build_neighbor_lists") == 0) {

                categorySums["neighbors"] += counterValue;

            } else {
                if(counterName.length() > 0) {
                    categorySums[counterName] += counterValue;
                } else {
                    categorySums["other"] += counterValue;
                }
            }
        }

        // Print the accumulated sums for each category
        for(const auto& category: categorySums) {
            std::cout << category.first << ": " << category.second << std::endl;
        }
    }

private:
    std::vector<std::string> counter_names;
    std::vector<TimeType> counters;
    std::vector<std::chrono::high_resolution_clock::time_point> clocks;
    TimeType time_factor;
};

}
