if(PAIRS_BUILD_WITH_WALBERLA)
    pairs_generate_lib( 
        GEN_LIB     reduction_example 
        SCRIPT      force_reduction.py
    )
    add_executable( force_reduction force_reduction.cpp )
    target_link_libraries( force_reduction reduction_example )
endif()