if(PAIRS_BUILD_WITH_WALBERLA)
    pairs_generate_lib( 
        GEN_LIB  sd 
        SCRIPT   sd.py 
    )
    add_executable( sd_static sd_static.cpp )
    target_link_libraries( sd_static sd )

    pairs_generate_lib( 
        GEN_LIB  sd_no_pbc 
        SCRIPT   sd_no_pbc.py
    )
    add_executable( sd_static_triangular sd_static_triangular.cpp )
    target_link_libraries( sd_static_triangular sd_no_pbc )
endif()