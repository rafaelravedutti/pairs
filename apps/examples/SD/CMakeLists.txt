if(PAIRS_BUILD_WITH_WALBERLA)
    # Generate the library once and later attach it to different executables
    pairs_generate_lib( 
        GEN_LIB spring_dashpot 
        SCRIPT spring_dashpot.py )

    add_executable( sd_1 sd_1.cpp )
    target_link_libraries( sd_1 spring_dashpot )

    add_executable(sd_2 sd_2.cpp)
    target_link_libraries( sd_2 spring_dashpot )

    add_executable(sd_3_CPU sd_3_CPU.cpp)
    target_link_libraries( sd_3_CPU spring_dashpot )

    add_executable(sd_4 sd_4.cpp)
    target_link_libraries( sd_4 spring_dashpot )
    
    if(PAIRS_BUILD_WITH_CUDA)
        add_executable(sd_3_GPU sd_3_GPU.cu)
        target_link_libraries( sd_3_GPU spring_dashpot )
    endif()
endif()