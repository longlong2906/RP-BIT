# Opt-in safety-constrained surface length. Legacy planner policies stay intact.
target_sources(terrain25d PRIVATE src/Safety.cpp)
add_executable(safety_tests tests/unit/safety_tests.cpp)
target_link_libraries(safety_tests PRIVATE terrain25d)
add_test(NAME safe_length_unit COMMAND safety_tests)
add_executable(safety_check apps/safety_check.cpp)
target_link_libraries(safety_check PRIVATE terrain25d)
add_test(NAME safe_length_field COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_safe_length_field.py)
add_test(NAME safe_length_runner COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_safe_length_runner.py)
add_test(NAME safe_length_oracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_safe_length_oracle.py
  --cli $<TARGET_FILE:safety_check>)
set_tests_properties(safe_length_unit safe_length_field safe_length_runner safe_length_oracle
  PROPERTIES LABELS safe_length)
if(TERRAIN_WITH_OMPL)
  add_executable(safe_length_ompl_tests tests/integration/safe_length_ompl_tests.cpp)
  target_link_libraries(safe_length_ompl_tests PRIVATE terrain_ompl)
  add_test(NAME safe_length_ompl COMMAND safe_length_ompl_tests)
  set_tests_properties(safe_length_ompl PROPERTIES LABELS safe_length)
  if(WIN32)
    set_tests_properties(safe_length_ompl PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
endif()
