# Query-calibrated hard safety domain; no changes to native OMPL.
target_sources(terrain25d PRIVATE src/RiskCalibration.cpp)
add_executable(risk_calibration_tests tests/unit/risk_calibration_tests.cpp)
target_link_libraries(risk_calibration_tests PRIVATE terrain25d)
add_test(NAME safe_length_v2_calibration COMMAND risk_calibration_tests)
foreach(component protocol runner validation analysis)
  add_test(NAME safe_length_v2_${component} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_safe_length_v2_${component}.py)
  set_tests_properties(safe_length_v2_${component} PROPERTIES LABELS safe_length_v2)
endforeach()
set_tests_properties(safe_length_v2_calibration PROPERTIES LABELS safe_length_v2)
add_test(NAME safe_length_v2_oracle_roundoff COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_SOURCE_DIR}/tests/test_oracle_clipping_roundoff.py)
set_tests_properties(safe_length_v2_oracle_roundoff PROPERTIES LABELS safe_length_v2)
if(TERRAIN_WITH_OMPL)
  add_test(NAME safe_length_v2_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_safe_length_v2_worker.py
    --worker $<TARGET_FILE:terrain_benchmark> --safety-cli $<TARGET_FILE:safety_check>)
  set_tests_properties(safe_length_v2_worker PROPERTIES LABELS safe_length_v2)
endif()
if(WIN32)
  get_property(v2_tests DIRECTORY PROPERTY TESTS)
  foreach(name IN LISTS v2_tests)
    if(name MATCHES "^safe_length_v2_")
      set_tests_properties(${name} PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin;PATH=path_list_prepend:C:/Users/Long/anaconda3/Library/bin")
    endif()
  endforeach()
endif()
