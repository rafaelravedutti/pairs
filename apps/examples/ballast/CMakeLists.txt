pairs_generate_lib( 
        GEN_LIB  ballastGen 
        SCRIPT   ballast.py
)
add_executable( ballast ballast.cpp )
target_link_libraries( ballast ballastGen )