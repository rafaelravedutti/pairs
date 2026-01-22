if(PAIRS_BUILD_WITH_WALBERLA)
    pairs_generate_lib( sd sd.py gen_dir_pbc )
    add_executable( sd_static sd_static.cpp )
    pairs_attach_to_target( sd_static sd )

    pairs_generate_lib( sd_no_pbc sd_no_pbc.py gen_dir_no_pbc )
    add_executable( sd_static_triangular sd_static_triangular.cpp )
    pairs_attach_to_target( sd_static_triangular sd_no_pbc )
endif()