if(PAIRS_BUILD_WITH_WALBERLA)
    pairs_generate_lib( 
        GEN_LIB  global_example 
        SCRIPT   sphere_box_global.py
    )
    add_executable( sphere_box_global sphere_box_global.cpp )
    target_link_libraries( sphere_box_global global_example )
endif()
