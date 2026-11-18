# Whole-path exact-PRI optimizer around one persistent OMPL BIT* instance.
if(TERRAIN_WITH_OMPL)
  target_sources(terrain_ompl PRIVATE src/PriPathSearch.cpp)
  add_executable(pri_pathsearch_tests tests/integration/pri_pathsearch_tests.cpp)
  target_link_libraries(pri_pathsearch_tests PRIVATE terrain_ompl)
  add_test(NAME pri_pathsearch_core COMMAND pri_pathsearch_tests)
  set_tests_properties(pri_pathsearch_core PROPERTIES LABELS pri_pathsearch TIMEOUT 180)
  if(WIN32)
    set_tests_properties(pri_pathsearch_core PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
  add_test(NAME pri_pathsearch_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_pri_pathsearch_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(pri_pathsearch_worker PROPERTIES LABELS pri_pathsearch RUN_SERIAL TRUE TIMEOUT 600)
endif()
foreach(component protocol runner validation analysis)
  add_test(NAME pri_pathsearch_${component} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_pri_pathsearch_${component}.py)
  set_tests_properties(pri_pathsearch_${component} PROPERTIES LABELS pri_pathsearch RUN_SERIAL TRUE TIMEOUT 600)
endforeach()
