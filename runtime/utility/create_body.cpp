#include "create_body.hpp"

namespace pairs {

// returns the uid of the body created, or 0 if the body is not created
id_t create_halfspace(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double nx, double ny, double nz, 
                    int type, int flag){
    // TODO: increase capacity if exceeded
    id_t uid = 0;
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));   
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto normals = pr->getAsVectorProperty(pr->getPropertyByName("normal"));

    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z) || flag & (flags::INFINITE | flags::GLOBAL) ){
        int n = pr->getTrackedVariableAsInteger("nlocal");
        uid = (flag & (flags::INFINITE | flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
        uids(n) = uid;
        positions(n, 0) = x;
        positions(n, 1) = y;
        positions(n, 2) = z;
        normals(n, 0) = nx;
        normals(n, 1) = ny;
        normals(n, 2) = nz;
        types(n) = type;
        flags(n) = flag;
        shapes(n) = Shapes::Halfspace;
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }

    return uid;
}

// returns the uid of the body created, or 0 if the body is not created
id_t create_sphere(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double vx, double vy, double vz, 
                    double density, double radius, int type, int flag){
    // TODO: increase capacity if exceeded
    id_t uid = 0;
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));   
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));
    auto radii = pr->getAsFloatProperty(pr->getPropertyByName("radius"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto velocities = pr->getAsVectorProperty(pr->getPropertyByName("linear_velocity"));
    auto angular_velocity = pr->getAsVectorProperty(pr->getPropertyByName("angular_velocity"));

    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z) || flag & (flags::INFINITE | flags::GLOBAL)) {
        int n = pr->getTrackedVariableAsInteger("nlocal");
        uid = (flag & (flags::INFINITE | flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
        uids(n) = uid;
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
        shapes(n) = Shapes::Sphere;
        angular_velocity(n, 0) = 0;
        angular_velocity(n, 1) = 0;
        angular_velocity(n, 2) = 0;
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }
    
    return uid;
}

id_t create_box(PairsRuntime *pr, 
    double x, double y, double z, 
    double vx, double vy, double vz, 
    double ex, double ey, double ez, 
    double density, int type, int flag){
    // TODO: increase capacity if exceeded
    id_t uid = 0;
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));   
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto velocities = pr->getAsVectorProperty(pr->getPropertyByName("linear_velocity"));
    auto edge_length = pr->getAsVectorProperty(pr->getPropertyByName("edge_length"));
    auto angular_velocity = pr->getAsVectorProperty(pr->getPropertyByName("angular_velocity"));

    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z) || flag & (flags::INFINITE | flags::GLOBAL)) {
        int n = pr->getTrackedVariableAsInteger("nlocal");
        uid = (flag & (flags::INFINITE | flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
        uids(n) = uid;
        edge_length(n, 0) = ex;
        edge_length(n, 1) = ey;
        edge_length(n, 2) = ez;
        masses(n) = ex * ey * ez * density;
        positions(n, 0) = x;
        positions(n, 1) = y;
        positions(n, 2) = z;
        velocities(n, 0) = vx;
        velocities(n, 1) = vy;
        velocities(n, 2) = vz;
        types(n) = type;
        flags(n) = flag;
        shapes(n) = Shapes::Box;
        angular_velocity(n, 0) = 0;
        angular_velocity(n, 1) = 0;
        angular_velocity(n, 2) = 0;
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }

    return uid;
}

id_t create_clump(PairsRuntime *pr, 
    double x, double y, double z, 
    double vx, double vy, double vz, 
    double density, double radius, int type, int flag){
    // TODO: increase capacity if exceeded
    id_t uid = 0;
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));   
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));
    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto velocities = pr->getAsVectorProperty(pr->getPropertyByName("linear_velocity"));
    auto angular_velocity = pr->getAsVectorProperty(pr->getPropertyByName("angular_velocity"));

    real_t * local_positions = static_cast<real_t *>((pr->getArrayByName("local_positions")).getHostPointer());
    real_t * local_radius = static_cast<real_t *>((pr->getArrayByName("local_radius")).getHostPointer());

    double r = 0.05;
    local_radius[0] = r;
    local_radius[1] = r;
    local_radius[2] = r;
    double sq3 = r * sqrt(3.0)/3.0;
    double pos0[3] = {-r,   -sq3,   0};
    double pos1[3] = {r,    -sq3,    0};
    double pos2[3] = {0,    2*sq3,  0};

    local_positions[3*0 + 0] = pos0[0]; 
    local_positions[3*0 + 1] = pos0[1]; 
    local_positions[3*0 + 2] = pos0[2]; 

    local_positions[3*1 + 0] = pos1[0]; 
    local_positions[3*1 + 1] = pos1[1]; 
    local_positions[3*1 + 2] = pos1[2]; 

    local_positions[3*2 + 0] = pos2[0]; 
    local_positions[3*2 + 1] = pos2[1]; 
    local_positions[3*2 + 2] = pos2[2]; 

    double  total_mass  = 4.0 / 3.0 * M_PI * local_radius[0] * local_radius[0] * local_radius[0] * density;
            total_mass += 4.0 / 3.0 * M_PI * local_radius[1] * local_radius[1] * local_radius[1] * density;
            total_mass += 4.0 / 3.0 * M_PI * local_radius[2] * local_radius[2] * local_radius[2] * density;


    if(pr->getDomainPartitioner()->isWithinSubdomain(x, y, z) || flag & (flags::INFINITE | flags::GLOBAL)) {
        int n = pr->getTrackedVariableAsInteger("nlocal");
        uid = (flag & (flags::INFINITE | flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
        uids(n) = uid;
        masses(n) = total_mass;
        
        positions(n, 0) = x;
        positions(n, 1) = y;
        positions(n, 2) = z;
        velocities(n, 0) = vx;
        velocities(n, 1) = vy;
        velocities(n, 2) = vz;
        types(n) = type;
        flags(n) = flag;
        shapes(n) = Shapes::Clump;
        angular_velocity(n, 0) = 0;
        angular_velocity(n, 1) = 0;
        angular_velocity(n, 2) = 0;
        pr->setTrackedVariableAsInteger("nlocal", n + 1);
    }

    return uid;
}


}