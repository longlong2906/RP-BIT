if(TERRAIN_WITH_OMPL)
  target_sources(terrain_ompl PRIVATE src/PriSegmentSearch.cpp)
  add_executable(pri_segment_search_tests tests/integration/pri_segment_search_tests.cpp)
  target_link_libraries(pri_segment_search_tests PRIVATE terrain_ompl)
  add_test(NAME pri_segment_core COMMAND pri_segment_search_tests)
  set_tests_properties(pri_segment_core PROPERTIES LABELS pri_segment TIMEOUT 180)
  if(WIN32)
    set_tests_properties(pri_segment_core PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
  add_test(NAME pri_segment_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_pri_segment_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(pri_segment_worker PROPERTIES LABELS pri_segment RUN_SERIAL TRUE TIMEOUT 600)
endif()
add_test(NAME pri_segment_protocol COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_SOURCE_DIR}/tests/test_pri_segment_protocol.py)
set_tests_properties(pri_segment_protocol PROPERTIES LABELS pri_segment RUN_SERIAL TRUE TIMEOUT 600)
