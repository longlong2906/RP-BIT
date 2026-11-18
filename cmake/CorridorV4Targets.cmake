foreach(component protocol validation runner)
  add_test(NAME corridor_v4_${component} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_corridor_v4_${component}.py)
  set_tests_properties(corridor_v4_${component} PROPERTIES LABELS corridor_v4)
endforeach()
if(TERRAIN_WITH_OMPL)
  add_test(NAME corridor_v4_fixed_work COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_corridor_v4_fixed_work.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(corridor_v4_fixed_work PROPERTIES LABELS corridor_v4 TIMEOUT 360 RUN_SERIAL TRUE)
endif()
