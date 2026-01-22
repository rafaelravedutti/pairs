pairs_generate_lib( linear_spring_dashpot linear_spring_dashpot.py gen_dir )
add_executable( lsd_1 lsd_1.cpp )
pairs_attach_to_target( lsd_1 linear_spring_dashpot )