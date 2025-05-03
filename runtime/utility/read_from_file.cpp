#include "read_from_file.hpp"


namespace pairs {
void write_boxes(PairsRuntime *pr, const char *filename){
// std::cout << "wite boxes =========== " << std::endl;
    int nlocal = pr->getTrackedVariableAsInteger("nlocal");
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));

    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto edge_lengths = pr->getAsVectorProperty(pr->getPropertyByName("edge_length"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File out_file;
    int error_code = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out_file);

    std::ostringstream out_stream;

    if (error_code != MPI_SUCCESS) {
        std::cerr << "Failed to open " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }

    for(int n=0; n<nlocal; ++n){
        if(shapes(n) == Shapes::Box){
            if(flags(n) & (flags::INFINITE | flags::GLOBAL)){
                if(pr->getDomainPartitioner()->getRank()==0){
                    out_stream << shapes(n) << " " << flags(n) << " " << types(n) << " " <<
                                positions(n, 0) << " " << positions(n,1) << " " << positions(n,2) << " " <<
                                edge_lengths(n, 0) << " " << edge_lengths(n, 1) << " " << edge_lengths(n, 2) << " "<< masses(n) << "\n";
                }
            }
            else{
                out_stream << shapes(n) << " " << flags(n) << " " << types(n) << " " <<
                            positions(n, 0) << " " << positions(n,1) << " " << positions(n,2) << " " <<
                            edge_lengths(n, 0) << " " << edge_lengths(n, 1) << " " << edge_lengths(n, 2) << " "<< masses(n) << "\n";
            }

        }
    }
    
    std::string output = out_stream.str();
    MPI_File_write_ordered(out_file, output.c_str(), (int)(output.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&out_file);
    MPI_Barrier(MPI_COMM_WORLD);
}


void read_boxes(PairsRuntime *pr, const char *filename){
    int n = pr->getTrackedVariableAsInteger("nlocal");
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));

    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto edge_lengths = pr->getAsVectorProperty(pr->getPropertyByName("edge_length"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));

    std::ifstream in_file(filename);

    if(!in_file.is_open()) {
        std::cerr << "Error: Could not open file \"" << filename << "\"" << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(in_file, line)) {
        std::istringstream in_stream(line);
        int shape;
        int flag;
        int type;
        double x, y, z;
        double edge_length[3];
        double mass;
        in_stream >> shape >> flag >> type >> x >> y >> z >> edge_length[0] >> edge_length[1] >> edge_length[2] >> mass;

        if(shape == Shapes::Box){
            bool within_subdom = pr->getDomainPartitioner()->isWithinSubdomain(x, y, z);
            if(within_subdom || (flag & (flags::INFINITE | flags::GLOBAL))){
                shapes(n) = shape;
                flags(n) = flag;
                types(n) = type;
                uids(n) = (flag & (flags::INFINITE |flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
                positions(n,0) = x;
                positions(n,1) = y;
                positions(n,2) = z;
                edge_lengths(n,0) = edge_length[0];
                edge_lengths(n,1) = edge_length[1];
                edge_lengths(n,2) = edge_length[2];
                masses(n) = mass;
                ++n;
            }
        }
    }

    pr->setTrackedVariableAsInteger("nlocal", n);

}

void write_spheres(PairsRuntime *pr, const char *filename){
    // std::cout << "wite spheres =========== " << std::endl;

    int nlocal = pr->getTrackedVariableAsInteger("nlocal");
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));

    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto radii = pr->getAsFloatProperty(pr->getPropertyByName("radius"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File out_file;
    int error_code = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out_file);

    std::ostringstream out_stream;

    if (error_code != MPI_SUCCESS) {
        std::cerr << "Failed to open " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }

    for(int n=0; n<nlocal; ++n){
        if(shapes(n) == Shapes::Sphere){
            out_stream << shapes(n) << " " << flags(n) << " " << types(n) << " " <<
                        positions(n, 0) << " " << positions(n,1) << " " << positions(n,2) << " " <<
                        radii(n) << " " << masses(n) << "\n";

        }
    }
    
    std::string output = out_stream.str();
    MPI_File_write_ordered(out_file, output.c_str(), (int)(output.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&out_file);
    MPI_Barrier(MPI_COMM_WORLD);
}

void read_spheres(PairsRuntime *pr, const char *filename, std::array<double, 3> offset){
    int n = pr->getTrackedVariableAsInteger("nlocal");
    auto shapes = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto flags = pr->getAsIntegerProperty(pr->getPropertyByName("flags"));
    auto types = pr->getAsIntegerProperty(pr->getPropertyByName("type"));
    auto uids = pr->getAsUInt64Property(pr->getPropertyByName("uid"));

    auto positions = pr->getAsVectorProperty(pr->getPropertyByName("position"));
    auto radii = pr->getAsFloatProperty(pr->getPropertyByName("radius"));
    auto masses = pr->getAsFloatProperty(pr->getPropertyByName("mass"));

    std::ifstream in_file(filename);

    if(!in_file.is_open()) {
        std::cerr << "Error: Could not open file \"" << filename << "\"" << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(in_file, line)) {
        std::istringstream in_stream(line);
        int shape;
        int flag;
        int type;
        double x, y, z;
        double radius;
        double mass;
        in_stream >> shape >> flag >> type >> x >> y >> z >> radius >> mass;

        double shifted_posx = x + offset[0];
        double shifted_posy = y + offset[1];
        double shifted_posz = z + offset[2];

        if(shape == Shapes::Sphere){
            bool within_subdom = pr->getDomainPartitioner()->isWithinSubdomain(shifted_posx, shifted_posy, shifted_posz);
            if(within_subdom || (flag & (flags::INFINITE | flags::GLOBAL))){
                shapes(n) = shape;
                flags(n) = flag;
                types(n) = type;
                uids(n) = (flag & (flags::INFINITE |flags::GLOBAL)) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
                positions(n,0) = shifted_posx;
                positions(n,1) = shifted_posy;
                positions(n,2) = shifted_posz;
                radii(n) = radius;
                masses(n) = mass;
                ++n;
            }
        }
    }

    pr->setTrackedVariableAsInteger("nlocal", n);

}

/*
void read_grid_data(PairsRuntime *pr, const char *filename, real_t *grid_buffer) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;

    if(!in_file.is_open()) {
        std::cerr << "Error: Could not open file \"" << filename << "\"" << std::endl;
        exit(-1);
    }

    std::getline(in_file, line);
    std::stringstream line_stream(line);
    std::string in0;
    int i = 0;

    while(std::getline(line_stream, in0, ',')) {
        //PAIRS_ASSERT(i < ndims * 2);
        grid_buffer[i] = std::stod(in0);
        i++;
    }

    in_file.close();
}
*/

size_t read_particle_data(
    PairsRuntime *pr, const char *filename, const property_t properties[],
    int shape_id, int start) {

    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;
    auto shape_ptr = pr->getAsIntegerProperty(pr->getPropertyByName("shape"));
    auto uid_ptr = pr->getAsUInt64Property(pr->getPropertyByName("uid"));
    int n = start;

    if(!in_file.is_open()) {
        std::cerr << "Error: Could not open file \"" << filename << "\"" << std::endl;
        exit(-1);
    }

    while(std::getline(in_file, line)) {
        std::stringstream line_stream(line);
        std::string in0;
        int within_domain = 1;
        int i = 0;
        int flags = 0;

        while(std::getline(line_stream, in0, ',')) {
            property_t p_id = properties[i];
            auto prop = pr->getProperty(p_id);
            auto prop_type = prop.getType();

            if(prop_type == Prop_Vector) {
                auto vector_ptr = pr->getAsVectorProperty(prop);
                std::string in1, in2;
                std::getline(line_stream, in1, ',');
                std::getline(line_stream, in2, ',');
                real_t x = std::stod(in0);
                real_t y = std::stod(in1);
                real_t z = std::stod(in2);
                vector_ptr(n, 0) = x;
                vector_ptr(n, 1) = y;
                vector_ptr(n, 2) = z;

                if(prop.getName() == "position") {
                    within_domain = pr->getDomainPartitioner()->isWithinSubdomain(x, y, z);
                }
            } else if(prop_type == Prop_Matrix) {
                auto matrix_ptr = pr->getAsMatrixProperty(prop);
                constexpr int nelems = 9;
                std::string in_buf;

                matrix_ptr(n, 0) = std::stod(in0);
                for(int e = 1; e < nelems; e++) {
                    std::getline(line_stream, in_buf, ',');
                    matrix_ptr(n, e) = std::stod(in_buf);
                }
            } else if(prop_type == Prop_Quaternion) {
                auto quat_ptr = pr->getAsQuaternionProperty(prop);
                constexpr int nelems = 4;
                std::string in_buf;

                quat_ptr(n, 0) = std::stod(in0);
                for(int e = 1; e < nelems; e++) {
                    std::getline(line_stream, in_buf, ',');
                    quat_ptr(n, e) = std::stod(in_buf);
                }
            } else if(prop_type == Prop_Integer) {
                auto int_ptr = pr->getAsIntegerProperty(prop);
                int_ptr(n) = std::stoi(in0);

                if(prop.getName() == "flags") {
                    flags = int_ptr(n);
                }
            } else if(prop_type == Prop_UInt64) {
                auto uint64_ptr = pr->getAsUInt64Property(prop);
                uint64_ptr(n) = std::stoi(in0);

                if(prop.getName() == "uid") {
                    std::cerr << "Can't read uid from file." << std::endl;
                    exit(-1);
                }
            } else if(prop_type == Prop_Real) {
                auto float_ptr = pr->getAsFloatProperty(prop);
                float_ptr(n) = std::stod(in0);
            } else {
                std::cerr << "read_particle_data(): Invalid property type!" << std::endl;
                return 0;
            }

            i++;
        }

        if(within_domain || flags & (flags::INFINITE | flags::GLOBAL)) {
            uid_ptr(n) = (flags & flags::GLOBAL) ? UniqueID::createGlobal(pr) : UniqueID::create(pr);
            shape_ptr(n++) = shape_id;
        }
    }

    return n;
}

/*
size_t read_feature_data(PairsRuntime *pr, const char *filename, const int feature_id, const property_t properties[], size_t nprops) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;

    if(in_file.is_open()) {
        while(std::getline(in_file, line)) {
            std::stringstream line_stream(line);
            std::string istr, jstr, in0;
            std::getline(line_stream, istr, ',');
            std::getline(line_stream, jstr, ',');
            int i = std::stoi(istr);
            int j = std::stoi(jstr);

            while(std::getline(line_stream, in0, ',')) {
                property_t p_id = properties[i];
                auto prop = pr->getProperty(p_id);
                auto prop_type = prop.getType();

                if(prop_type == Prop_Vector) {
                    auto vector_ptr = pr->getAsVectorFeatureProperty(prop);
                    std::string in1, in2;
                    std::getline(line_stream, in1, ',');
                    std::getline(line_stream, in2, ',');
                    real_t x = std::stod(in0);
                    real_t y = std::stod(in1);
                    real_t z = std::stod(in2);
                    vector_ptr(i, j, 0) = x;
                    vector_ptr(i, j, 1) = y;
                    vector_ptr(i, j, 2) = z;
                } else if(prop_type == Prop_Integer) {
                    auto int_ptr = pr->getAsIntegerFeatureProperty(prop);
                    int_ptr(i, j) = std::stoi(in0);
                } else if(prop_type == Prop_Real) {
                    auto float_ptr = pr->getAsFloatFeatureProperty(prop);
                    float_ptr(i, j) = std::stod(in0);
                } else {
                    std::cerr << "read_feature_data(): Invalid property type!" << std::endl;
                    return 0;
                }
            }
        }

        in_file.close();
    }

    return n;
}
*/

}
