# if(PAIRS_BUILD_WITH_WALBERLA)
    # Generate the library once and later attach it to different executables
    pairs_generate_lib( spring_dashpot spring_dashpot.py gen_dir )

    add_executable( sd_1 sd_1.cpp )
    pairs_attach_to_target( sd_1 spring_dashpot )

    add_executable(sd_2 sd_2.cpp)
    pairs_attach_to_target( sd_2 spring_dashpot )

    add_executable(sd_3_CPU sd_3_CPU.cpp)
    pairs_attach_to_target( sd_3_CPU spring_dashpot )

    add_executable(sd_4 sd_4.cpp)
    pairs_attach_to_target( sd_4 spring_dashpot )
    
    if(PAIRS_BUILD_WITH_CUDA)
        add_executable(sd_3_GPU sd_3_GPU.cu)
        pairs_attach_to_target( sd_3_GPU spring_dashpot )
    endif()
# endif()