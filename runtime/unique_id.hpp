#include "pairs.hpp"

#pragma once

namespace pairs {

class UniqueID{
public:
    inline static id_t create(PairsRuntime *pr);
    inline static id_t createGlobal(PairsRuntime *pr);

private:
    static const id_t capacity = 1000000000;   // max number of particles per rank
    inline static id_t counter = 1;
    inline static id_t globalCounter = 1;

};

inline id_t UniqueID::create(PairsRuntime *pr){
    id_t rank = static_cast<id_t>(pr->getDomainPartitioner()->getRank());
    id_t id = rank*capacity + counter;
    ++counter;
    return id;
}

inline id_t UniqueID::createGlobal(PairsRuntime *pr){
    id_t numranks = static_cast<id_t>(pr->getDomainPartitioner()->getWorldSize());
    id_t id = numranks*capacity + globalCounter;
    ++globalCounter;
    return id;
}

}
