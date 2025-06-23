#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <mpi.h>


template<typename Type>
void print_global_stats(std::string name, Type value, MPI_Datatype mpi_type, MPI_Comm comm){
    int rank, world_size;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    std::vector<Type> all_values(world_size);
    MPI_Gather(&value, 1, mpi_type, all_values.data(), 1, mpi_type, 0, comm);

    if(rank == 0){
        std::sort(all_values.begin(), all_values.end());
        Type sum = std::accumulate(all_values.begin(), all_values.end(), Type());
        double avg = static_cast<double>(sum)/world_size;

        // Standard deviation ------------------
        double sum_sq = 0.0;
        for(const auto &v : all_values){
            sum_sq += (static_cast<double>(v) - avg) * (static_cast<double>(v) - avg);
        }
        double std_dev = std::sqrt(sum_sq / world_size);

        // Median ------------------------------
        double median = 0.0;
        if(world_size%2 == 0){
            median = static_cast<double>(all_values[world_size/2 - 1] + all_values[world_size/2]) / 2.0;   
        } else{
            median = static_cast<double>(all_values[world_size/2]);
        }

        std::cout << "-----------------------------------" << std::endl;
        std::cout << name + "_MIN: " << all_values[0] << std::endl;
        std::cout << name + "_MAX: " << all_values[world_size - 1] << std::endl;
        std::cout << name + "_SUM: " << sum << std::endl;
        std::cout << name + "_AVG: " << avg << std::endl;
        std::cout << name + "_MED: " << median << std::endl;
        std::cout << name + "_STDDEV: " << std_dev << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
