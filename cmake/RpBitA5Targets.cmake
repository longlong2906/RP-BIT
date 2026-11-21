if(TERRAIN_WITH_OMPL)
  target_sources(terrain_ompl PRIVATE src/RpBitA5.cpp)
  add_executable(rp_bit_a5_tests tests/integration/rp_bit_a5_tests.cpp)
  target_link_libraries(rp_bit_a5_tests PRIVATE terrain_ompl)
  add_test(NAME rp_bit_a5_core COMMAND rp_bit_a5_tests)
  set_tests_properties(rp_bit_a5_core PROPERTIES LABELS rp_bit TIMEOUT 600)
  if(WIN32)
    set_tests_properties(rp_bit_a5_core PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
  add_test(NAME rp_bit_a5_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_rp_bit_a5_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(rp_bit_a5_worker PROPERTIES LABELS rp_bit RUN_SERIAL TRUE TIMEOUT 900)
endif()
