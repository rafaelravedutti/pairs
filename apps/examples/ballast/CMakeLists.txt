pairs_generate_lib( ballastGen ballast.py gen_dir )
add_executable( ballast ballast.cpp )
pairs_attach_to_target( ballast ballastGen )