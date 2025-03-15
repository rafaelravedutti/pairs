#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <iomanip>

#pragma once

using namespace std;

namespace pairs {

template<typename TimeType, typename TimeUnit = std::chrono::nanoseconds>
class Timers {
public:
    Timers(TimeType _factor) : time_factor(_factor) {}
    ~Timers() {}

    void add(size_t id, std::string name) {
        counter_names.resize(id + 1);
        time_counters.resize(id + 1);
        call_counters.resize(id + 1);
        clocks.resize(id + 1);
        counter_names[id] = name;
    }

    void start(size_t id) { clocks[id] = std::chrono::high_resolution_clock::now(); ++call_counters[id];}

    void stop(size_t id) {
        auto current_clock = std::chrono::high_resolution_clock::now();
        time_counters[id] += static_cast<TimeType>(
            std::chrono::duration_cast<TimeUnit>(current_clock - clocks[id]).count()) * time_factor;
    }

    void writeToFile(int rank, int world_size){
        std::string filename = "timers_" + std::to_string(world_size) + ".txt";
        if (rank==0) std::cout << "Writing timers log to: " << filename << std::endl;

        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);

        std::ostringstream ss;
        ss << "Rank: " << rank << "\n";
        ss << std::left << std::setw(80) << "Timer"
           << std::left << std::setw(15) << "Total [ms]"
           << std::left << std::setw(15) << "Count" << "\n";
        ss << "--------------------------------------------------------------------------------------------------------\n";
        
        // Modules
        for (size_t i = TimerMarkers::Offset; i < time_counters.size(); ++i) {
            const std::string& counterName = counter_names[i];
            if(counterName.length() > 0) {
                ss << std::left << std::setw(80) << counter_names[i]
                    << std::left << std::setw(15) << std::fixed << std::setprecision(2) << time_counters[i]
                    << std::left << std::setw(15) << call_counters[i]
                    << "\n";
            }
        }

        // Markers
        for (size_t i = 0; i < TimerMarkers::Offset; ++i) {
            ss << std::left << std::setw(80) << counter_names[i]
                << std::left << std::setw(15) << std::fixed << std::setprecision(2) << time_counters[i]
                << std::left << std::setw(15) << 1
                << "\n";
        }

        computeCategories();

        // Categories
        for (const auto& cs : categorySums) {;
            ss << std::left << std::setw(80) << cs.first
                << std::left << std::setw(15) << std::fixed << std::setprecision(2) << cs.second
                << std::left << std::setw(15) << 1
                << "\n";
        }
        ss << "\n\n";

        std::string output = ss.str();
        MPI_File_write_ordered(file, output.c_str(), output.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&file);
    }

    void print(){
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(80) << "Timer (MPI rank: 0)"
            << std::left << std::setw(15) << "Total [ms]"
            << std::left << std::setw(15) << "Count" << "\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        
        // Modules
        for (size_t i = TimerMarkers::Offset; i < time_counters.size(); ++i) {
            const std::string& counterName = counter_names[i];
            // if(counterName.find("INTERFACE_MODULES::") == 0) {
            if(counterName.length() > 0) {
                std::cout << std::left << std::setw(80) << counter_names[i]
                        << std::left << std::setw(15) << std::fixed << std::setprecision(2) << time_counters[i]
                        << std::left << std::setw(15) << call_counters[i]
                        << "\n";
            }
        }

        // Markers
        for (size_t i = 0; i < TimerMarkers::Offset; ++i) {
            std::cout << std::left << std::setw(80) << counter_names[i]
                    << std::left << std::setw(15) << std::fixed << std::setprecision(2) << time_counters[i]
                    << std::left << std::setw(15) << 1
                    << "\n";
        }

        computeCategories();
        
        // Categories
        for (const auto& cs : categorySums) {;
            std::cout << std::left << std::setw(80) << cs.first
                    << std::left << std::setw(15) << std::fixed << std::setprecision(2) << cs.second
                    << std::left << std::setw(15) << 1
                    << "\n";
        }
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
    }

    void computeCategories() {
        categorySums.clear();
        for (size_t i = 0; i < time_counters.size(); ++i) {
            const std::string& counterName = counter_names[i];
            TimeType counterValue = time_counters[i];

            if(counterName.find("INTERNAL_MODULES::pack_") == 0 ||
               counterName.find("INTERNAL_MODULES::unpack_") == 0 ||
               counterName.find("INTERNAL_MODULES::determine_") == 0 ||
               counterName.find("INTERNAL_MODULES::set_communication_") == 0 ||
               counterName.find("INTERNAL_MODULES::remove_exchanged_particles") == 0 ||
               counterName.find("INTERNAL_MODULES::change_size_after_exchange") == 0) {

                categorySums["INTERNAL_CATEGORIES::COMMUNICATION"] += counterValue;

            } else if(counterName.find("INTERNAL_MODULES::build_cell_lists") == 0 ||
                      counterName.find("INTERNAL_MODULES::build_cell_lists_stencil") == 0 ||
                      counterName.find("INTERNAL_MODULES::partition_cell_lists") == 0 ||
                      counterName.find("INTERNAL_MODULES::build_neighbor_lists") == 0) {

                categorySums["INTERNAL_CATEGORIES::NEIGHBORS"] += counterValue;
            }
        }
    }

private:
    std::vector<std::string> counter_names;
    std::vector<TimeType> time_counters;
    std::vector<int> call_counters;
    std::unordered_map<std::string, TimeType> categorySums;
    std::vector<std::chrono::high_resolution_clock::time_point> clocks;
    TimeType time_factor;
};

}
