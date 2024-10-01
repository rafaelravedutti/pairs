#include "create_shape.hpp"

namespace pairs {

void create_halfspace(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double nx, double ny, double nz, 
                    int type, int flag){
    // TODO: ensure unique id in all functions that create particle or read particle from file
    // TODO: increase capacity if exceeded
    // auto uids = pr->getAsIntegerProperty(pr->getPropertyByName("uid"));   
    auto shape = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto normals = pr->getAsVectorProperty(pr->getPropertyByName("normal"));

    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z) || flag & (FLAGS_INFINITE | FLAGS_FIXED | FLAGS_GLOBAL) ){
        int n = pr->getTrackedVariableAsInteger("nlocal");
        // uids(n) = ;
        positions(n, 0) = x;
        positions(n, 1) = y;
        positions(n, 2) = z;
        normals(n, 0) = nx;
        normals(n, 1) = ny;
        normals(n, 2) = nz;
        types(n) = type;
        flags(n) = flag;
        shape(n) = 1;   // halfspace
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }
}

void create_particle(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double vx, double vy, double vz, 
                    double density, double radius, int type, int flag){
    // TODO: ensure unique id in all functions that create particle or read particle from file
    // TODO: increase capacity if exceeded
    // auto uids = pr->getAsIntegerProperty(pr->getPropertyByName("uid"));   
    auto shape = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));
    auto radii = pr->getAsFloatProperty(pr->getPropertyByName("radius"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto velocities = pr->getAsVectorProperty(pr->getPropertyByName("linear_velocity"));

    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z)) {
        int n = pr->getTrackedVariableAsInteger("nlocal");
        // uids(n) = ;
        radii(n) = radius;
        masses(n) = ((4.0 / 3.0) * M_PI) * radius * radius * radius * density;
        positions(n, 0) = x;
        positions(n, 1) = y;
        positions(n, 2) = z;
        velocities(n, 0) = vx;
        velocities(n, 1) = vy;
        velocities(n, 2) = vz;
        types(n) = type;
        flags(n) = flag;
        shape(n) = 0;   // sphere
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }
}

}