# Schema 11 is opt-in at runtime; all legacy worker defaults remain unchanged.
add_executable(putn_paper_tests tests/test_putn_paper.cpp)
target_link_libraries(putn_paper_tests PRIVATE putn_static)
add_test(NAME putn_paper COMMAND putn_paper_tests)
foreach(name native_session_policy native_paper_protocol native_paper_analysis native_paper_pipeline native_paper_oracle)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_${name}.py)
endforeach()
if(TERRAIN_COMPARE_SGP_PYTHON)
  add_test(NAME sem_native_planner COMMAND ${TERRAIN_COMPARE_SGP_PYTHON}
    ${CMAKE_SOURCE_DIR}/tests/test_sem_native_planner.py)
  add_test(NAME native_models COMMAND ${TERRAIN_COMPARE_SGP_PYTHON}
    ${CMAKE_SOURCE_DIR}/tests/test_native_models.py --library $<TARGET_FILE:terrain_compare_c>)
  add_test(NAME native_paper_worker COMMAND ${TERRAIN_COMPARE_SGP_PYTHON}
    ${CMAKE_SOURCE_DIR}/tests/test_native_paper_worker.py)
  add_test(NAME sgp_prediction_cache COMMAND ${TERRAIN_COMPARE_SGP_PYTHON}
    ${CMAKE_SOURCE_DIR}/tests/test_sgp_prediction_cache.py)
  add_test(NAME sgp_float32_policy COMMAND ${TERRAIN_COMPARE_SGP_PYTHON}
    ${CMAKE_SOURCE_DIR}/tests/test_sgp_float32_policy.py)
else()
  add_test(NAME sem_native_planner COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_sem_native_planner.py)
endif()
