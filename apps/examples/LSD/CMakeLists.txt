pairs_generate_lib( 
        GEN_LIB  linear_spring_dashpot
        SCRIPT   linear_spring_dashpot.py 
)

add_executable( lsd_1 lsd_1.cpp )
target_link_libraries( lsd_1 linear_spring_dashpot )